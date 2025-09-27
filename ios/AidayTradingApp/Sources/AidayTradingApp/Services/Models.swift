import Foundation

struct AuthTokens: Codable {
    let accessToken: String
    let refreshToken: String
    let accessTokenExpiry: Date

    var isAccessTokenValid: Bool {
        accessTokenExpiry > Date()
    }
}

struct UserProfile: Codable, Identifiable {
    enum Role: String, Codable {
        case viewer
        case admin
    }

    let id: UUID
    let username: String
    let email: String
    let role: Role
    let approvalStatus: ApprovalStatus

    enum ApprovalStatus: String, Codable {
        case pending
        case approved
        case rejected
    }
}

struct UserSessionContext {
    let profile: UserProfile
    let tokens: AuthTokens
}

struct APIErrorResponse: Codable, Error {
    let message: String
}
