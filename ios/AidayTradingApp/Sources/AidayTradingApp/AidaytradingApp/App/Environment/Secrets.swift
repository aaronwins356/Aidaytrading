import Foundation

enum Secrets {
    static func jwtToken() throws -> String {
        guard let tokens = try KeychainStorage().load() else {
            throw SecretsError.missingToken
        }
        return tokens.accessToken
    }
}

enum SecretsError: Error {
    case missingToken
}
