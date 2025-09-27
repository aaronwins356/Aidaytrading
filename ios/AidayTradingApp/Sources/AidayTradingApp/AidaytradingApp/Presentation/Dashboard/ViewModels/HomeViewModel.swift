import Foundation

@MainActor
final class HomeViewModel: ObservableObject {
    @Published var isLoading = false
    @Published var status: Status?
    @Published var profit: ProfitSnapshot?
    @Published var equity: [EquityPoint] = []
    @Published var error: String?
    @Published var lastUpdated: Date?
    @Published var isStale = false

    private let repository: DashboardRepository
    private var pollingTask: Task<Void, Never>?

    init(repository: DashboardRepository = DashboardRepositoryImpl()) {
        self.repository = repository
        bootstrapFromCache()
    }

    deinit {
        pollingTask?.cancel()
    }

    func loadInitial() async {
        isLoading = true
        error = nil
        do {
            async let statusTask = repository.fetchStatus()
            async let profitTask = repository.fetchProfit()
            async let equityTask = repository.fetchEquityCurve()
            let (status, profit, equity) = try await (statusTask, profitTask, equityTask)
            self.status = status
            self.profit = profit
            self.equity = equity
            self.lastUpdated = Date()
            self.isStale = false
        } catch {
            self.error = error.localizedDescription
            self.isStale = true
        }
        isLoading = false
    }

    func startPolling() {
        guard pollingTask == nil else { return }
        pollingTask = Task { [weak self] in
            guard let self else { return }
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: UInt64(AppConfig.pollingInterval * 1_000_000_000))
                await self.refreshSilently()
            }
        }
    }

    func stopPolling() {
        pollingTask?.cancel()
        pollingTask = nil
    }

    func equityDelta() -> (amount: Decimal, percent: Decimal)? {
        guard let first = equity.first?.equity, let last = equity.last?.equity else { return nil }
        let delta = last - first
        let percent = first == 0 ? 0 : (delta / first) * 100
        return (delta, percent)
    }

    private func bootstrapFromCache() {
        if let cachedStatus = repository.cachedStatus() {
            status = cachedStatus
        }
        if let cachedProfit = repository.cachedProfit() {
            profit = cachedProfit
        }
        if let cachedEquity = repository.cachedEquityCurve() {
            equity = cachedEquity
        }
    }

    private func refreshSilently() async {
        do {
            async let statusTask = repository.fetchStatus()
            async let profitTask = repository.fetchProfit()
            async let equityTask = repository.fetchEquityCurve()
            let (newStatus, newProfit, newEquity) = try await (statusTask, profitTask, equityTask)
            status = newStatus
            profit = newProfit
            equity = newEquity
            lastUpdated = Date()
            isStale = false
            error = nil
        } catch {
            self.error = error.localizedDescription
            isStale = true
        }
    }
}
