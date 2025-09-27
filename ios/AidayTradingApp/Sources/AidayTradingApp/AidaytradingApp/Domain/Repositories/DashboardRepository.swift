import Foundation

protocol DashboardRepository {
    func fetchStatus() async throws -> Status
    func fetchProfit() async throws -> ProfitSnapshot
    func fetchEquityCurve() async throws -> [EquityPoint]
    func cachedStatus() -> Status?
    func cachedProfit() -> ProfitSnapshot?
    func cachedEquityCurve() -> [EquityPoint]?
}
