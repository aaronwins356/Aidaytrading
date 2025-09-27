import Foundation

protocol ApprovalServiceProtocol {
    func fetchStatus(username: String, email: String) async throws -> UserProfile.ApprovalStatus
}

struct ApprovalService: ApprovalServiceProtocol {
    private let apiClient: APIClientProtocol

    init(apiClient: APIClientProtocol = APIClient()) {
        self.apiClient = apiClient
    }

    func fetchStatus(username: String, email: String) async throws -> UserProfile.ApprovalStatus {
        let response: ApprovalStatusResponse = try await apiClient.send(
            ApprovalRequest.status(username: username, email: email),
            decode: ApprovalStatusResponse.self
        )
        return response.status
    }
}
