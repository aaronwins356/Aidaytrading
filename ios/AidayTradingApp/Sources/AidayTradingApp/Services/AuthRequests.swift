import Foundation

enum AuthRequest: APIRequestConvertible {
    case signup(username: String, email: String, password: String)
    case login(email: String, password: String)
    case refresh(refreshToken: String)
    case profile(accessToken: String)

    var urlRequest: URLRequest {
        get throws {
            var request: URLRequest
            switch self {
            case let .signup(username, email, password):
                let url = APIEnvironment.baseURL.appending(path: "/auth/signup")
                request = URLRequest(url: url)
                request.httpMethod = "POST"
                request.httpBody = try JSONEncoder().encode([
                    "username": username,
                    "email": email,
                    "password": password
                ])
            case let .login(email, password):
                let url = APIEnvironment.baseURL.appending(path: "/auth/login")
                request = URLRequest(url: url)
                request.httpMethod = "POST"
                request.httpBody = try JSONEncoder().encode([
                    "email": email,
                    "password": password
                ])
            case let .refresh(refreshToken):
                let url = APIEnvironment.baseURL.appending(path: "/auth/refresh")
                request = URLRequest(url: url)
                request.httpMethod = "POST"
                request.httpBody = try JSONEncoder().encode([
                    "refresh_token": refreshToken
                ])
            case let .profile(accessToken):
                let url = APIEnvironment.baseURL.appending(path: "/users/me")
                request = URLRequest(url: url)
                request.httpMethod = "GET"
                request.addValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
            }
            request.addValue("application/json", forHTTPHeaderField: "Content-Type")
            request.addValue("application/json", forHTTPHeaderField: "Accept")
            return request
        }
    }
}

struct AuthResponse: Decodable {
    let tokens: AuthTokens
    let user: UserProfile
}
