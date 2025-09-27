import Foundation

final class DashboardRepositoryImpl: DashboardRepository {
    private let apiClient: APIClientPerforming
    private let cache: DiskCache

    init(
        apiClient: APIClientPerforming = APIClient(),
        cache: DiskCache = .shared
    ) {
        self.apiClient = apiClient
        self.cache = cache
    }

    func fetchStatus() async throws -> Status {
        do {
            let dto: StatusDTO = try await apiClient.get(Endpoints.status())
            let status = dto.toEntity()
            try cache.save(status, for: "status_latest.json")
            return status
        } catch {
            if let cached = cachedStatus() {
                return cached
            }
            throw error
        }
    }

    func fetchProfit() async throws -> ProfitSnapshot {
        do {
            let dto: ProfitDTO = try await apiClient.get(Endpoints.profit())
            let snapshot = dto.toEntity()
            try cache.save(snapshot, for: "profit_latest.json")
            return snapshot
        } catch {
            if let cached = cachedProfit() {
                return cached
            }
            throw error
        }
    }

    func fetchEquityCurve() async throws -> [EquityPoint] {
        do {
            let dtos: [EquityPointDTO] = try await apiClient.get(Endpoints.equityCurve())
            let points = dtos.map { $0.toEntity() }
            try cache.save(points, for: "equity_latest.json")
            return downsample(points)
        } catch {
            if let cached = cachedEquityCurve() {
                return downsample(cached)
            }
            throw error
        }
    }

    func cachedStatus() -> Status? {
        cache.load(Status.self, for: "status_latest.json")
    }

    func cachedProfit() -> ProfitSnapshot? {
        cache.load(ProfitSnapshot.self, for: "profit_latest.json")
    }

    func cachedEquityCurve() -> [EquityPoint]? {
        cache.load([EquityPoint].self, for: "equity_latest.json")
    }

    private func downsample(_ points: [EquityPoint], targetCount: Int = 300) -> [EquityPoint] {
        guard points.count > targetCount else { return points }
        let stride = max(points.count / targetCount, 1)
        return points.enumerated().compactMap { index, point in
            index % stride == 0 ? point : nil
        }
    }
}
