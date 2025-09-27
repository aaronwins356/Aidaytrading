import Foundation

struct AdminEndpoint {
    enum Method: String {
        case get = "GET"
        case post = "POST"
        case patch = "PATCH"
    }

    let path: String
    var method: Method = .get
    var body: Data?

    func urlRequest(baseURL: URL) throws -> URLRequest {
        guard baseURL.scheme?.lowercased() == "https" else {
            throw HTTPError.nonHTTPS
        }
        guard let url = URL(string: path, relativeTo: baseURL) else {
            throw URLError(.badURL)
        }
        var request = URLRequest(url: url)
        request.httpMethod = method.rawValue
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if let body {
            request.httpBody = body
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        return request
    }
}

enum AdminEndpoints {
    static func users() -> AdminEndpoint {
        AdminEndpoint(path: "/admin/users")
    }

    static func updateUser(id: UUID, body: Data) -> AdminEndpoint {
        AdminEndpoint(path: "/admin/users/\(id.uuidString)", method: .patch, body: body)
    }

    static func resetPassword(id: UUID) -> AdminEndpoint {
        AdminEndpoint(path: "/admin/users/\(id.uuidString)/reset-password", method: .post)
    }

    static func risk() -> AdminEndpoint {
        AdminEndpoint(path: "/admin/risk")
    }

    static func updateRisk(body: Data) -> AdminEndpoint {
        AdminEndpoint(path: "/admin/risk", method: .patch, body: body)
    }

    static func botStatus() -> AdminEndpoint {
        AdminEndpoint(path: "/admin/bot/status")
    }

    static func botStart() -> AdminEndpoint {
        AdminEndpoint(path: "/admin/bot/start", method: .post)
    }

    static func botStop() -> AdminEndpoint {
        AdminEndpoint(path: "/admin/bot/stop", method: .post)
    }

    static func botMode(body: Data) -> AdminEndpoint {
        AdminEndpoint(path: "/admin/bot/mode", method: .patch, body: body)
    }
}
