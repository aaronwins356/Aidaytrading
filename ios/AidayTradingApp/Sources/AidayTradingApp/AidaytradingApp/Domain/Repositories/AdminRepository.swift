import Foundation

protocol AdminRepository {
    func fetchRiskConfiguration() async throws -> RiskConfiguration
    func updateRiskConfiguration(_ configuration: RiskConfiguration) async throws -> RiskConfiguration
    func fetchUsers() async throws -> [AdminUser]
    func updateUser(id: UUID, role: AdminUser.Role?, status: AdminUser.Status?) async throws -> AdminUser
    func resetPassword(id: UUID) async throws
    func fetchBotStatus() async throws -> BotStatus
    func startBot() async throws -> BotStatus
    func stopBot() async throws -> BotStatus
    func setBotMode(_ mode: BotMode) async throws -> BotStatus
}

final class AdminRepositoryImpl: AdminRepository {
    private let api: AdminAPIServiceProtocol
    private let encoder: JSONEncoder

    init(api: AdminAPIServiceProtocol = AdminAPIService()) {
        self.api = api
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.dateEncodingStrategy = .iso8601
        self.encoder = encoder
    }

    func fetchRiskConfiguration() async throws -> RiskConfiguration {
        try await api.fetch(AdminEndpoints.risk(), as: RiskConfiguration.self)
    }

    func updateRiskConfiguration(_ configuration: RiskConfiguration) async throws -> RiskConfiguration {
        let body = try encoder.encode(configuration.clamped())
        return try await api.fetch(AdminEndpoints.updateRisk(body: body), as: RiskConfiguration.self)
    }

    func fetchUsers() async throws -> [AdminUser] {
        try await api.fetch(AdminEndpoints.users(), as: [AdminUser].self)
    }

    func updateUser(id: UUID, role: AdminUser.Role?, status: AdminUser.Status?) async throws -> AdminUser {
        let payload = UpdateUserPayload(role: role, status: status)
        let body = try encoder.encode(payload)
        return try await api.fetch(AdminEndpoints.updateUser(id: id, body: body), as: AdminUser.self)
    }

    func resetPassword(id: UUID) async throws {
        try await api.send(AdminEndpoints.resetPassword(id: id))
    }

    func fetchBotStatus() async throws -> BotStatus {
        let dto = try await api.fetch(AdminEndpoints.botStatus(), as: BotStatusDTO.self)
        return dto.toDomain()
    }

    func startBot() async throws -> BotStatus {
        let dto = try await api.fetch(AdminEndpoints.botStart(), as: BotStatusDTO.self)
        return dto.toDomain()
    }

    func stopBot() async throws -> BotStatus {
        let dto = try await api.fetch(AdminEndpoints.botStop(), as: BotStatusDTO.self)
        return dto.toDomain()
    }

    func setBotMode(_ mode: BotMode) async throws -> BotStatus {
        let payload = BotModePayload(paperTrading: mode.isPaperTrading)
        let body = try encoder.encode(payload)
        let dto = try await api.fetch(AdminEndpoints.botMode(body: body), as: BotStatusDTO.self)
        return dto.toDomain()
    }
}

private struct UpdateUserPayload: Encodable {
    let role: AdminUser.Role?
    let status: AdminUser.Status?
}

private struct BotModePayload: Encodable {
    let paperTrading: Bool
}

private struct BotStatusDTO: Decodable {
    let running: Bool
    let paperTrading: Bool
    let updatedAt: Date?

    func toDomain() -> BotStatus {
        BotStatus(running: running, mode: BotMode(paperTrading: paperTrading), lastUpdated: updatedAt ?? Date())
    }
}
