import Foundation
import SwiftUI

enum TradeOutcomeFilter: String, CaseIterable, Identifiable {
    case all
    case winners
    case losers

    var id: String { rawValue }

    var title: String {
        switch self {
        case .all:
            return "All"
        case .winners:
            return "Wins"
        case .losers:
            return "Losses"
        }
    }
}

@MainActor
final class TradesViewModel: ObservableObject {
    @Published var trades: [TradeRecord] = []
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var hasMorePages = true
    @Published var symbolFilter: String = ""
    @Published var outcomeFilter: TradeOutcomeFilter = .all
    @Published var realtimeWarningMessage: String?

    private let accessToken: String
    private let reportingService: ReportingServiceProtocol
    private var currentPage = 1
    private let pageSize = 50
    private let realtimeService: RealtimeServiceProtocol
    private let notificationScheduler: LocalNotificationScheduling

    init(
        accessToken: String,
        reportingService: ReportingServiceProtocol,
        realtimeService: RealtimeServiceProtocol = RealtimeService(),
        notificationScheduler: LocalNotificationScheduling = LocalNotificationScheduler()
    ) {
        self.accessToken = accessToken
        self.reportingService = reportingService
        self.realtimeService = realtimeService
        self.notificationScheduler = notificationScheduler
        self.realtimeService.delegate = self
    }

    func loadInitial() async {
        currentPage = 1
        hasMorePages = true
        trades = []
        realtimeService.disconnect()
        realtimeService.connect(accessToken: accessToken)
        repeat {
            await loadMore()
        } while outcomeFilter != .all && filteredTrades.isEmpty && hasMorePages
    }

    func loadMore() async {
        guard hasMorePages, !isLoading else { return }
        isLoading = true
        errorMessage = nil
        do {
            let response = try await reportingService.fetchTrades(
                accessToken: accessToken,
                page: currentPage,
                pageSize: pageSize,
                symbol: symbolFilter.isEmpty ? nil : symbolFilter,
                side: nil,
                start: nil,
                end: nil
            )
            currentPage += 1
            trades.append(contentsOf: response.items)
            hasMorePages = trades.count < response.total
        } catch {
            errorMessage = (error as? APIErrorResponse)?.message ?? error.localizedDescription
            hasMorePages = false
        }
        isLoading = false
    }

    var filteredTrades: [TradeRecord] {
        trades.filter { trade in
            switch outcomeFilter {
            case .all:
                return true
            case .winners:
                return trade.pnl >= 0
            case .losers:
                return trade.pnl < 0
            }
        }
    }

    func stop() {
        realtimeService.disconnect()
        notificationScheduler.cancelRealtimeStallNotification()
    }

    deinit {
        realtimeService.disconnect()
    }
}

extension TradesViewModel: RealtimeServiceDelegate {
    func realtimeServiceDidConnect(_ service: RealtimeServiceProtocol) {
        realtimeWarningMessage = nil
        notificationScheduler.cancelRealtimeStallNotification()
    }

    func realtimeService(_ service: RealtimeServiceProtocol, didDisconnectWith error: Error?) {
        realtimeWarningMessage = "Realtime feed disconnected."
    }

    func realtimeService(_ service: RealtimeServiceProtocol, didReceiveStatus status: SystemStatus) {
        // No-op for the trades ledger.
    }

    func realtimeService(_ service: RealtimeServiceProtocol, didReceiveEquityPoint point: EquityCurvePoint) {
        // No-op for the trades ledger.
    }

    func realtimeService(_ service: RealtimeServiceProtocol, didReceiveTrade trade: TradeRecord) {
        if !trades.contains(where: { $0.id == trade.id }) {
            trades.insert(trade, at: 0)
        }
        notificationScheduler.cancelRealtimeStallNotification()
    }

    func realtimeServiceDidDetectStall(_ service: RealtimeServiceProtocol) {
        realtimeWarningMessage = "Realtime feed disconnected."
        notificationScheduler.cancelRealtimeStallNotification()
        Task {
            await notificationScheduler.scheduleRealtimeStallNotification()
        }
    }
}
