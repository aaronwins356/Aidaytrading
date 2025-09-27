import XCTest
@testable import AidayTradingApp

final class RealtimeServiceTests: XCTestCase {
    func testReconnectsAfterFailure() async throws {
        let statusMessage = try XCTUnwrap("{" +
            "\"running\":true,\"uptime_seconds\":120" +
            "}".data(using: .utf8))
        let equityMessage = try JSONSerialization.data(withJSONObject: [
            ISO8601DateFormatter.apiFormatter.string(from: Date()),
            "1000.00"
        ])
        let tradeMessage = try JSONSerialization.data(withJSONObject: [
            "id": 1,
            "symbol": "BTCUSD",
            "side": "buy",
            "size": "1.0",
            "pnl": "50.00",
            "timestamp": ISO8601DateFormatter.apiFormatter.string(from: Date())
        ])

        let factory = MockWebSocketClientFactory()
        factory.enqueue(
            client: MockWebSocketClient(channel: .status, events: [.message(statusMessage), .failure(MockError.notConfigured)]),
            for: .status
        )
        factory.enqueue(
            client: MockWebSocketClient(channel: .status, events: [.message(statusMessage)]),
            for: .status
        )
        factory.enqueue(client: MockWebSocketClient(channel: .equity, events: [.message(equityMessage)]), for: .equity)
        factory.enqueue(client: MockWebSocketClient(channel: .equity, events: [.message(equityMessage)]), for: .equity)
        factory.enqueue(client: MockWebSocketClient(channel: .trades, events: [.message(tradeMessage)]), for: .trades)
        factory.enqueue(client: MockWebSocketClient(channel: .trades, events: [.message(tradeMessage)]), for: .trades)

        let service = RealtimeService(factory: factory, stallInterval: 0.5, initialReconnectDelay: 0.05)
        let delegate = CapturingRealtimeDelegate()
        service.delegate = delegate

        service.connect(accessToken: "token")

        try await Task.sleep(nanoseconds: 200_000_000)
        XCTAssertGreaterThanOrEqual(delegate.connectedCount, 1)

        try await Task.sleep(nanoseconds: 700_000_000)
        XCTAssertTrue(delegate.didReceiveDisconnect)
        XCTAssertGreaterThanOrEqual(delegate.connectedCount, 2)
    }
}

private final class CapturingRealtimeDelegate: RealtimeServiceDelegate {
    private(set) var connectedCount = 0
    private(set) var didReceiveDisconnect = false

    func realtimeServiceDidConnect(_ service: RealtimeServiceProtocol) {
        connectedCount += 1
    }

    func realtimeService(_ service: RealtimeServiceProtocol, didDisconnectWith error: Error?) {
        didReceiveDisconnect = true
    }

    func realtimeService(_ service: RealtimeServiceProtocol, didReceiveStatus status: SystemStatus) {}
    func realtimeService(_ service: RealtimeServiceProtocol, didReceiveEquityPoint point: EquityCurvePoint) {}
    func realtimeService(_ service: RealtimeServiceProtocol, didReceiveTrade trade: TradeRecord) {}
    func realtimeServiceDidDetectStall(_ service: RealtimeServiceProtocol) {}
}

private final class MockWebSocketClientFactory: WebSocketClientFactory {
    private var queues: [WebSocketChannel: [MockWebSocketClient]] = [:]

    func enqueue(client: MockWebSocketClient, for channel: WebSocketChannel) {
        var queue = queues[channel] ?? []
        queue.append(client)
        queues[channel] = queue
    }

    func makeClient(url: URL, headers: [String: String], channel: WebSocketChannel, delegate: WebSocketClientDelegate?) -> WebSocketClient {
        guard var queue = queues[channel], !queue.isEmpty else {
            fatalError("No mock client configured for channel \(channel)")
        }
        let client = queue.removeFirst()
        queues[channel] = queue
        client.delegate = delegate
        return client
    }
}

private final class MockWebSocketClient: WebSocketClient {
    enum Event {
        case message(Data)
        case failure(Error)
    }

    weak var delegate: WebSocketClientDelegate?
    private let channel: WebSocketChannel
    private var events: [Event]

    init(channel: WebSocketChannel, events: [Event]) {
        self.channel = channel
        self.events = events
    }

    func connect() {
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.01) { [weak self] in
            guard let self else { return }
            self.delegate?.webSocketClientDidOpen(self, channel: self.channel)
            self.processNext()
        }
    }

    func disconnect(with code: URLSessionWebSocketTask.CloseCode) {
        events.removeAll()
    }

    private func processNext() {
        guard !events.isEmpty else { return }
        let event = events.removeFirst()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.01) { [weak self] in
            guard let self else { return }
            switch event {
            case .message(let data):
                self.delegate?.webSocketClient(self, channel: self.channel, didReceive: data)
                self.processNext()
            case .failure(let error):
                self.delegate?.webSocketClient(self, channel: self.channel, didFailWith: error)
            }
        }
    }
}
