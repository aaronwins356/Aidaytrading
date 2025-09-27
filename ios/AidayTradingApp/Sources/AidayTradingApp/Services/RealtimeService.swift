import Foundation
import os

protocol RealtimeServiceDelegate: AnyObject {
    func realtimeServiceDidConnect(_ service: RealtimeServiceProtocol)
    func realtimeService(_ service: RealtimeServiceProtocol, didDisconnectWith error: Error?)
    func realtimeService(_ service: RealtimeServiceProtocol, didReceiveStatus status: SystemStatus)
    func realtimeService(_ service: RealtimeServiceProtocol, didReceiveEquityPoint point: EquityCurvePoint)
    func realtimeService(_ service: RealtimeServiceProtocol, didReceiveTrade trade: TradeRecord)
    func realtimeServiceDidDetectStall(_ service: RealtimeServiceProtocol)
}

protocol RealtimeServiceProtocol: AnyObject {
    var delegate: RealtimeServiceDelegate? { get set }
    func connect(accessToken: String)
    func disconnect()
}

protocol WebSocketClient: AnyObject {
    var delegate: WebSocketClientDelegate? { get set }
    func connect()
    func disconnect(with code: URLSessionWebSocketTask.CloseCode)
}

enum WebSocketChannel: String {
    case status
    case equity
    case trades
}

protocol WebSocketClientDelegate: AnyObject {
    func webSocketClientDidOpen(_ client: WebSocketClient, channel: WebSocketChannel)
    func webSocketClient(_ client: WebSocketClient, channel: WebSocketChannel, didReceive data: Data)
    func webSocketClient(_ client: WebSocketClient, channel: WebSocketChannel, didFailWith error: Error)
}

protocol WebSocketClientFactory {
    func makeClient(url: URL, headers: [String: String], channel: WebSocketChannel, delegate: WebSocketClientDelegate?) -> WebSocketClient
}

final class URLSessionWebSocketClient: NSObject, WebSocketClient {
    private let task: URLSessionWebSocketTask
    private let channel: WebSocketChannel
    weak var delegate: WebSocketClientDelegate?

    init(session: URLSession, request: URLRequest, channel: WebSocketChannel, delegate: WebSocketClientDelegate?) {
        self.task = session.webSocketTask(with: request)
        self.channel = channel
        self.delegate = delegate
        super.init()
    }

    func connect() {
        task.resume()
        delegate?.webSocketClientDidOpen(self, channel: channel)
        listen()
    }

    private func listen() {
        task.receive { [weak self] result in
            guard let self else { return }
            switch result {
            case let .success(message):
                switch message {
                case let .data(data):
                    delegate?.webSocketClient(self, channel: channel, didReceive: data)
                case let .string(string):
                    if let data = string.data(using: .utf8) {
                        delegate?.webSocketClient(self, channel: channel, didReceive: data)
                    }
                @unknown default:
                    break
                }
                self.listen()
            case let .failure(error):
                delegate?.webSocketClient(self, channel: channel, didFailWith: error)
            }
        }
    }

    func disconnect(with code: URLSessionWebSocketTask.CloseCode) {
        task.cancel(with: code, reason: nil)
    }
}

struct URLSessionWebSocketClientFactory: WebSocketClientFactory {
    private let session: URLSession

    init(session: URLSession = .shared) {
        self.session = session
    }

    func makeClient(url: URL, headers: [String: String], channel: WebSocketChannel, delegate: WebSocketClientDelegate?) -> WebSocketClient {
        var request = URLRequest(url: url)
        headers.forEach { key, value in
            request.addValue(value, forHTTPHeaderField: key)
        }
        return URLSessionWebSocketClient(session: session, request: request, channel: channel, delegate: delegate)
    }
}

final class RealtimeService: NSObject, RealtimeServiceProtocol {
    weak var delegate: RealtimeServiceDelegate?

    private struct ChannelState {
        var client: WebSocketClient?
        var reconnectDelay: TimeInterval
        var isConnected = false
    }

    private var channels: [WebSocketChannel: ChannelState] = [:]
    private let factory: WebSocketClientFactory
    private var accessToken: String?
    private let decoder: JSONDecoder
    private let log = Logger(subsystem: "com.aidaytrading.app", category: "RealtimeService")
    private var stallTask: Task<Void, Never>?
    private let stallInterval: TimeInterval
    private let initialReconnectDelay: TimeInterval
    private var lastEquityUpdate: Date?

    init(
        factory: WebSocketClientFactory = URLSessionWebSocketClientFactory(),
        stallInterval: TimeInterval = 300,
        initialReconnectDelay: TimeInterval = 1
    ) {
        self.factory = factory
        self.stallInterval = stallInterval
        self.initialReconnectDelay = initialReconnectDelay
        decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        super.init()
    }

    func connect(accessToken: String) {
        self.accessToken = accessToken
        open(channel: .status, path: "/ws/status")
        open(channel: .equity, path: "/ws/equity")
        open(channel: .trades, path: "/ws/trades")
        resetStallMonitor()
    }

    func disconnect() {
        stallTask?.cancel()
        stallTask = nil
        channels.forEach { _, state in
            state.client?.disconnect(with: .goingAway)
        }
        channels.removeAll()
        accessToken = nil
        lastEquityUpdate = nil
    }

    private func open(channel: WebSocketChannel, path: String) {
        guard let token = accessToken else { return }
        let url = APIEnvironment.baseURL.appending(path: path)
        let headers = [
            "Authorization": "Bearer \(token)",
            "Accept": "application/json"
        ]
        var state = channels[channel] ?? ChannelState(client: nil, reconnectDelay: initialReconnectDelay, isConnected: false)
        let client = factory.makeClient(url: url, headers: headers, channel: channel, delegate: self)
        state.client = client
        state.isConnected = false
        state.reconnectDelay = initialReconnectDelay
        channels[channel] = state
        client.connect()
    }

    private func scheduleReconnect(for channel: WebSocketChannel) {
        guard var state = channels[channel] else { return }
        state.isConnected = false
        channels[channel] = state
        notifyDisconnected()
        stallTask?.cancel()
        resetStallMonitor()
        let delay = min(state.reconnectDelay, 60)
        log.debug("Scheduling reconnect for \(channel.rawValue, privacy: .public) in \(delay, privacy: .public)s")
        Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            await MainActor.run {
                guard let self else { return }
                var updatedState = self.channels[channel] ?? ChannelState(client: nil, reconnectDelay: self.initialReconnectDelay, isConnected: false)
                updatedState.reconnectDelay = min(delay * 2, 60)
                self.channels[channel] = updatedState
                self.open(channel: channel, path: "/ws/\(channel.rawValue)")
            }
        }
    }

    private func notifyConnectedIfNeeded() {
        if channels.values.allSatisfy({ $0.isConnected }) {
            delegate?.realtimeServiceDidConnect(self)
        }
    }

    private func notifyDisconnected() {
        delegate?.realtimeService(self, didDisconnectWith: nil)
    }

    private func resetStallMonitor() {
        stallTask?.cancel()
        stallTask = Task { [weak self] in
            guard let self else { return }
            try? await Task.sleep(nanoseconds: UInt64(stallInterval * 1_000_000_000))
            await MainActor.run {
                self.delegate?.realtimeServiceDidDetectStall(self)
            }
        }
    }

    private func handleEquityMessage(_ data: Data) {
        guard let point = try? decoder.decode(EquityCurvePoint.self, from: data) else { return }
        if let lastEquityUpdate, point.timestamp.timeIntervalSince(lastEquityUpdate) < 600 {
            return
        }
        lastEquityUpdate = point.timestamp
        delegate?.realtimeService(self, didReceiveEquityPoint: point)
    }
}

extension RealtimeService: WebSocketClientDelegate {
    func webSocketClientDidOpen(_ client: WebSocketClient, channel: WebSocketChannel) {
        guard var state = channels[channel] else { return }
        state.isConnected = true
        state.reconnectDelay = initialReconnectDelay
        channels[channel] = state
        notifyConnectedIfNeeded()
    }

    func webSocketClient(_ client: WebSocketClient, channel: WebSocketChannel, didReceive data: Data) {
        resetStallMonitor()
        switch channel {
        case .status:
            if let status = try? decoder.decode(SystemStatus.self, from: data) {
                delegate?.realtimeService(self, didReceiveStatus: status)
            }
        case .equity:
            handleEquityMessage(data)
        case .trades:
            if let trade = try? decoder.decode(TradeRecord.self, from: data) {
                delegate?.realtimeService(self, didReceiveTrade: trade)
            }
        }
    }

    func webSocketClient(_ client: WebSocketClient, channel: WebSocketChannel, didFailWith error: Error) {
        log.error("WebSocket channel \(channel.rawValue, privacy: .public) failed: \(error.localizedDescription, privacy: .public)")
        scheduleReconnect(for: channel)
    }
}
