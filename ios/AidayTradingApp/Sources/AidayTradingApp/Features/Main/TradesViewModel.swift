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

    private let accessToken: String
    private let reportingService: ReportingServiceProtocol
    private var currentPage = 1
    private let pageSize = 50

    init(accessToken: String, reportingService: ReportingServiceProtocol) {
        self.accessToken = accessToken
        self.reportingService = reportingService
    }

    func loadInitial() async {
        currentPage = 1
        hasMorePages = true
        trades = []
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
}
