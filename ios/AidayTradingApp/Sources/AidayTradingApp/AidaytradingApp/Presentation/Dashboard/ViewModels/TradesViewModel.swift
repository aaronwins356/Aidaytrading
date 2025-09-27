import Foundation

@MainActor
final class TradesViewModel: ObservableObject {
    enum Outcome: String, CaseIterable, Identifiable {
        case win
        case loss

        var id: String { rawValue }
    }

    struct TradesFilters: Equatable {
        var symbol: String?
        var outcome: Outcome?
        var startDate: Date?
        var endDate: Date?

        static let empty = TradesFilters()
    }

    @Published var isLoading = false
    @Published private(set) var items: [Trade] = []
    @Published private(set) var availableSymbols: [String] = []
    @Published var filters: TradesFilters = .empty
    @Published var error: String?
    @Published private(set) var nextCursor: String?

    private let repository: TradesRepository
    private var allTrades: [Trade] = []
    private var isLoadingMore = false
    private let pageSize = 50

    init(repository: TradesRepository = TradesRepositoryImpl()) {
        self.repository = repository
    }

    func loadFirstPage() async {
        isLoading = true
        error = nil
        do {
            let response = try await repository.fetchTradesPage(limit: pageSize, cursor: nil)
            allTrades = response.items
            nextCursor = response.nextCursor
            refreshSymbols()
            applyFilters()
        } catch {
            self.error = error.localizedDescription
        }
        isLoading = false
    }

    func loadNextPageIfNeeded(currentIndex: Int) async {
        guard let nextCursor, !isLoadingMore, currentIndex >= items.count - 5 else { return }
        isLoadingMore = true
        do {
            let response = try await repository.fetchTradesPage(limit: pageSize, cursor: nextCursor)
            nextCursor = response.nextCursor
            mergeTrades(response.items)
            refreshSymbols()
            applyFilters()
        } catch {
            self.error = error.localizedDescription
        }
        isLoadingMore = false
    }

    func apply(filters: TradesFilters) {
        self.filters = filters
        applyFilters()
    }

    func resetFilters() {
        filters = .empty
        applyFilters()
    }

    private func applyFilters() {
        var filtered = allTrades
        if let symbol = filters.symbol {
            filtered = filtered.filter { $0.symbol == symbol }
        }
        if let outcome = filters.outcome {
            filtered = filtered.filter { outcome == .win ? $0.isWin : !$0.isWin }
        }
        if let start = filters.startDate {
            filtered = filtered.filter { trade in
                guard let closedAt = trade.closedAt else { return false }
                return closedAt >= start
            }
        }
        if let end = filters.endDate {
            filtered = filtered.filter { trade in
                guard let closedAt = trade.closedAt else { return false }
                return closedAt <= end
            }
        }
        items = filtered.sorted { lhs, rhs in
            (lhs.closedAt ?? lhs.openedAt) > (rhs.closedAt ?? rhs.openedAt)
        }
    }

    private func mergeTrades(_ newTrades: [Trade]) {
        let existingIDs = Set(allTrades.map { $0.id })
        let filtered = newTrades.filter { !existingIDs.contains($0.id) }
        allTrades.append(contentsOf: filtered)
    }

    private func refreshSymbols() {
        let symbols = Set(allTrades.map { $0.symbol })
        availableSymbols = symbols.sorted()
    }
}
