import Foundation

enum ApprovalRequest: APIRequestConvertible {
    case status(username: String, email: String)

    var urlRequest: URLRequest {
        get throws {
            switch self {
            case let .status(username, email):
                var components = URLComponents(url: APIEnvironment.baseURL.appending(path: "/auth/status"), resolvingAgainstBaseURL: false)
                components?.queryItems = [
                    URLQueryItem(name: "username", value: username),
                    URLQueryItem(name: "email", value: email)
                ]
                guard let url = components?.url else {
                    throw URLError(.badURL)
                }
                var request = URLRequest(url: url)
                request.httpMethod = "GET"
                request.addValue("application/json", forHTTPHeaderField: "Accept")
                return request
            }
        }
    }
}

struct ApprovalStatusResponse: Decodable {
    let status: UserProfile.ApprovalStatus
}
