import Combine
import Foundation
import SwiftUI

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
    private var cancellables = Set<AnyCancellable>()
    private weak var realtimeClient: TradingWebSocketClientProtocol?

    init(repository: DashboardRepository = DashboardRepositoryImpl()) {
        self.repository = repository
        bootstrapFromCache()
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

    func attachRealtime(_ client: TradingWebSocketClientProtocol) {
        guard realtimeClient === nil else { return }
        realtimeClient = client

        client.equityPublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] point in
                guard let self else { return }
                if let last = self.equity.last, last.timestamp >= point.timestamp { return }
                withAnimation(.easeInOut(duration: 0.35)) {
                    self.equity.append(point)
                }
                self.lastUpdated = point.timestamp
                self.isStale = false
            }
            .store(in: &cancellables)

        client.statusPublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] status in
                guard let self else { return }
                self.status = status
                self.lastUpdated = Date()
                self.isStale = false
            }
            .store(in: &cancellables)
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
