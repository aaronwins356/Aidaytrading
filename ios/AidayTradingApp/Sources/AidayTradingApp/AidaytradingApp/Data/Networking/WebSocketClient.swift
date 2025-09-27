import Combine
import Foundation
import os

protocol TradingWebSocketClientProtocol: AnyObject {
    var equityPublisher: PassthroughSubject<EquityPoint, Never> { get }
    var tradePublisher: PassthroughSubject<Trade, Never> { get }
    var statusPublisher: PassthroughSubject<Status, Never> { get }
    var connectionState: CurrentValueSubject<TradingWebSocketClient.ConnectionState, Never> { get }

    func connect(accessToken: String)
    func disconnect()
    func sendPing()
}

final class TradingWebSocketClient: NSObject, TradingWebSocketClientProtocol, ObservableObject {
    enum ConnectionState: Equatable {
        case disconnected
        case connecting
        case connected
        case reconnecting(delay: TimeInterval)
    }

    struct Configuration {
        let url: URL
        let pingInterval: TimeInterval
        let decoder: JSONDecoder

        init(url: URL = AppConfig.webSocketURL, pingInterval: TimeInterval = 30, decoder: JSONDecoder = TradingWebSocketClient.makeDecoder()) {
            self.url = url
            self.pingInterval = pingInterval
            self.decoder = decoder
        }
    }

    let equityPublisher = PassthroughSubject<EquityPoint, Never>()
    let tradePublisher = PassthroughSubject<Trade, Never>()
    let statusPublisher = PassthroughSubject<Status, Never>()
    let connectionState = CurrentValueSubject<ConnectionState, Never>(.disconnected)

    private let configuration: Configuration
    private let log = Logger(subsystem: "com.aidaytrading.app", category: "TradingWebSocket")
    private lazy var urlSession: URLSession = {
        let configuration = URLSessionConfiguration.default
        configuration.waitsForConnectivity = true
        return URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
    }()

    private var webSocketTask: URLSessionWebSocketTask?
    private var receiveTask: Task<Void, Never>?
    private var pingTask: Task<Void, Never>?
    private var reconnectTask: Task<Void, Never>?
    private var reconnectDelay: TimeInterval = 1
    private var accessToken: String?
    private var lastEquityTimestamp: Date?
    private var processedTradeIDs = LRUSet<String>(capacity: 500)
    private var processedStatusTimestamps = LRUSet<Date>(capacity: 500)
    private let processingQueue = DispatchQueue(label: "com.aidaytrading.app.websocket")

    init(configuration: Configuration = Configuration()) {
        self.configuration = configuration
        super.init()
    }

    deinit {
        disconnect()
    }

    func connect(accessToken: String) {
        processingQueue.async { [weak self] in
            guard let self else { return }
            self.accessToken = accessToken
            self.cancelReconnect()
            self.openSocket()
        }
    }

    func disconnect() {
        processingQueue.async { [weak self] in
            guard let self else { return }
            self.accessToken = nil
            self.webSocketTask?.cancel(with: .goingAway, reason: nil)
            self.cleanupSocket()
            self.connectionState.send(.disconnected)
        }
    }

    func sendPing() {
        processingQueue.async { [weak self] in
            guard let self, let task = self.webSocketTask else { return }
            task.sendPing { error in
                if let error {
                    self.log.error("Ping failed: \(error.localizedDescription, privacy: .public)")
                }
            }
        }
    }

    private func openSocket() {
        guard let token = accessToken else { return }
        var request = URLRequest(url: configuration.url)
        request.timeoutInterval = configuration.pingInterval
        request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.addValue("application/json", forHTTPHeaderField: "Accept")
        connectionState.send(.connecting)
        webSocketTask = urlSession.webSocketTask(with: request)
        webSocketTask?.resume()
        scheduleReceiveLoop()
        schedulePing()
    }

    private func scheduleReceiveLoop() {
        receiveTask?.cancel()
        receiveTask = Task { [weak self] in
            guard let self else { return }
            await self.receiveMessages()
        }
    }

    private func schedulePing() {
        pingTask?.cancel()
        pingTask = Task { [weak self] in
            guard let self else { return }
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: UInt64(self.configuration.pingInterval * 1_000_000_000))
                self.sendPing()
            }
        }
    }

    private func receiveMessages() async {
        guard let task = webSocketTask else { return }
        while !Task.isCancelled {
            do {
                let message = try await task.receive()
                switch message {
                case let .data(data):
                    handleMessage(data)
                case let .string(text):
                    if let data = text.data(using: .utf8) {
                        handleMessage(data)
                    }
                @unknown default:
                    log.debug("Received unsupported message type")
                }
            } catch {
                log.error("Receive failed: \(error.localizedDescription, privacy: .public)")
                scheduleReconnect()
                return
            }
        }
    }

    private func handleMessage(_ data: Data) {
        processingQueue.async { [weak self] in
            guard let self else { return }
            do {
                let base = try self.configuration.decoder.decode(WebSocketEnvelope.self, from: data)
                switch base.type {
                case .equityUpdate:
                    let message = try self.configuration.decoder.decode(EquityUpdateMessage.self, from: data)
                    let timestamp = Date(timeIntervalSince1970: message.ts / 1000)
                    if let last = self.lastEquityTimestamp, timestamp <= last { return }
                    self.lastEquityTimestamp = timestamp
                    let point = EquityPoint(timestamp: timestamp, equity: Decimal(message.equity))
                    DispatchQueue.main.async {
                        self.equityPublisher.send(point)
                    }
                case .tradeExecuted:
                    let message = try self.configuration.decoder.decode(TradeExecutedMessage.self, from: data)
                    guard !self.processedTradeIDs.contains(message.id) else { return }
                    self.processedTradeIDs.insert(message.id)
                    let timestamp = Date(timeIntervalSince1970: message.ts / 1000)
                    let trade = Trade(
                        id: message.id,
                        symbol: message.symbol,
                        side: message.side,
                        quantity: Decimal(message.qty),
                        price: Decimal(message.price),
                        pnl: Decimal(message.pnl),
                        pnlPercent: Decimal(message.pnlPercent ?? 0),
                        openedAt: timestamp,
                        closedAt: timestamp,
                        timeZone: AppConfig.centralTimeZone
                    )
                    DispatchQueue.main.async {
                        self.tradePublisher.send(trade)
                    }
                case .botStatus:
                    let message = try self.configuration.decoder.decode(StatusMessage.self, from: data)
                    let timestamp = Date(timeIntervalSince1970: message.ts / 1000)
                    guard !self.processedStatusTimestamps.contains(timestamp) else { return }
                    self.processedStatusTimestamps.insert(timestamp)
                    let status = Status(running: message.running, uptimeSeconds: message.uptimeSeconds ?? 0)
                    DispatchQueue.main.async {
                        self.statusPublisher.send(status)
                    }
                }
            } catch {
                self.log.error("Failed to parse websocket payload: \(error.localizedDescription, privacy: .public)")
            }
        }
    }

    private func cleanupSocket(preserveBackoff: Bool = false) {
        receiveTask?.cancel()
        pingTask?.cancel()
        reconnectTask?.cancel()
        receiveTask = nil
        pingTask = nil
        reconnectTask = nil
        webSocketTask = nil
        lastEquityTimestamp = nil
        processedTradeIDs.removeAll()
        processedStatusTimestamps.removeAll()
        if !preserveBackoff {
            reconnectDelay = 1
        }
    }

    private func scheduleReconnect() {
        let delay = min(reconnectDelay, 60)
        cleanupSocket(preserveBackoff: true)
        guard accessToken != nil else { return }
        connectionState.send(.reconnecting(delay: delay))
        reconnectTask?.cancel()
        reconnectTask = Task { [weak self] in
            guard let self else { return }
            try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            self.processingQueue.async { [weak self] in
                guard let self, self.accessToken != nil else { return }
                self.reconnectDelay = min(delay * 2, 60)
                self.openSocket()
            }
        }
        reconnectDelay = min(delay * 2, 60)
    }

    private func cancelReconnect() {
        reconnectTask?.cancel()
        reconnectTask = nil
        reconnectDelay = 1
    }

    private static func makeDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .millisecondsSince1970
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }
}

extension TradingWebSocketClient: URLSessionWebSocketDelegate {
    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didOpenWithProtocol protocol: String?) {
        processingQueue.async { [weak self] in
            guard let self else { return }
            self.log.debug("WebSocket connected")
            self.reconnectDelay = 1
            self.connectionState.send(.connected)
        }
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        processingQueue.async { [weak self] in
            guard let self else { return }
            self.log.debug("WebSocket closed with code \(closeCode.rawValue, privacy: .public)")
            guard self.accessToken != nil else { return }
            self.scheduleReconnect()
        }
    }
}

private struct WebSocketEnvelope: Decodable {
    enum EventType: String, Decodable {
        case equityUpdate = "equity_update"
        case tradeExecuted = "trade_executed"
        case botStatus = "bot_status"
    }

    let type: EventType
}

private struct EquityUpdateMessage: Decodable {
    let type: WebSocketEnvelope.EventType
    let ts: TimeInterval
    let equity: Double
}

private struct TradeExecutedMessage: Decodable {
    let type: WebSocketEnvelope.EventType
    let id: String
    let symbol: String
    let side: String
    let qty: Double
    let price: Double
    let pnl: Double
    let pnlPercent: Double?
    let ts: TimeInterval
}

private struct StatusMessage: Decodable {
    let type: WebSocketEnvelope.EventType
    let running: Bool
    let ts: TimeInterval
    let uptimeSeconds: Int?
}

private struct LRUSet<Element: Hashable> {
    private var elements: [Element] = []
    private let capacity: Int

    init(capacity: Int) {
        self.capacity = max(capacity, 1)
    }

    mutating func insert(_ element: Element) {
        if let index = elements.firstIndex(of: element) {
            elements.remove(at: index)
        }
        elements.insert(element, at: 0)
        if elements.count > capacity {
            elements.removeLast()
        }
    }

    mutating func removeAll() {
        elements.removeAll(keepingCapacity: false)
    }

    func contains(_ element: Element) -> Bool {
        elements.contains(element)
    }
}
