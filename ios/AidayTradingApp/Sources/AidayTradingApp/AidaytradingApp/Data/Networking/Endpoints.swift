import Foundation

struct Endpoint {
    enum Method: String {
        case get = "GET"
    }

    let path: String
    var queryItems: [URLQueryItem] = []
    var method: Method = .get

    func urlRequest(baseURL: URL) throws -> URLRequest {
        guard baseURL.scheme?.lowercased() == "https" else {
            throw HTTPError.nonHTTPS
        }
        var components = URLComponents(url: baseURL.appendingPathComponent(path), resolvingAgainstBaseURL: false)
        if !queryItems.isEmpty {
            components?.queryItems = queryItems
        }
        guard let url = components?.url else {
            throw URLError(.badURL)
        }
        var request = URLRequest(url: url)
        request.httpMethod = method.rawValue
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        return request
    }
}

enum Endpoints {
    static func status() -> Endpoint {
        Endpoint(path: "/status")
    }

    static func profit() -> Endpoint {
        Endpoint(path: "/profit")
    }

    static func equityCurve() -> Endpoint {
        Endpoint(path: "/equity-curve")
    }

    static func trades(limit: Int, cursor: String?) -> Endpoint {
        var items: [URLQueryItem] = [URLQueryItem(name: "limit", value: String(limit))]
        if let cursor {
            items.append(URLQueryItem(name: "cursor", value: cursor))
        }
        return Endpoint(path: "/trades", queryItems: items)
    }
}
