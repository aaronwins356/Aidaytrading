import Foundation

enum ReportingRequest: APIRequestConvertible {
    case status(accessToken: String)
    case profit(accessToken: String)
    case balance(accessToken: String)
    case equityCurve(accessToken: String, start: Date?, end: Date?, limit: Int?)
    case trades(
        accessToken: String,
        page: Int,
        pageSize: Int,
        symbol: String?,
        side: TradeSide?,
        start: Date?,
        end: Date?
    )

    var urlRequest: URLRequest {
        get throws {
            let base = APIEnvironment.baseURL.appending(path: "/api/v1")
            let url: URL
            var request: URLRequest
            switch self {
            case let .status(accessToken):
                url = base.appending(path: "/status")
                request = URLRequest(url: url)
                request.httpMethod = "GET"
                request.addBearer(accessToken)
            case let .profit(accessToken):
                url = base.appending(path: "/profit")
                request = URLRequest(url: url)
                request.httpMethod = "GET"
                request.addBearer(accessToken)
            case let .balance(accessToken):
                url = base.appending(path: "/balance")
                request = URLRequest(url: url)
                request.httpMethod = "GET"
                request.addBearer(accessToken)
            case let .equityCurve(accessToken, start, end, limit):
                var components = URLComponents(url: base.appending(path: "/equity-curve"), resolvingAgainstBaseURL: false)!
                var queryItems: [URLQueryItem] = []
                if let start {
                    queryItems.append(URLQueryItem(name: "start", value: ISO8601DateFormatter.apiFormatter.string(from: start)))
                }
                if let end {
                    queryItems.append(URLQueryItem(name: "end", value: ISO8601DateFormatter.apiFormatter.string(from: end)))
                }
                if let limit {
                    queryItems.append(URLQueryItem(name: "limit", value: String(limit)))
                }
                components.queryItems = queryItems.isEmpty ? nil : queryItems
                guard let builtURL = components.url else {
                    throw URLError(.badURL)
                }
                request = URLRequest(url: builtURL)
                request.httpMethod = "GET"
                request.addBearer(accessToken)
            case let .trades(accessToken, page, pageSize, symbol, side, start, end):
                var components = URLComponents(url: base.appending(path: "/trades"), resolvingAgainstBaseURL: false)!
                var queryItems: [URLQueryItem] = [
                    URLQueryItem(name: "page", value: String(page)),
                    URLQueryItem(name: "page_size", value: String(pageSize))
                ]
                if let symbol, !symbol.isEmpty {
                    queryItems.append(URLQueryItem(name: "symbol", value: symbol.uppercased()))
                }
                if let side {
                    queryItems.append(URLQueryItem(name: "side", value: side.rawValue))
                }
                if let start {
                    queryItems.append(URLQueryItem(name: "start", value: ISO8601DateFormatter.apiFormatter.string(from: start)))
                }
                if let end {
                    queryItems.append(URLQueryItem(name: "end", value: ISO8601DateFormatter.apiFormatter.string(from: end)))
                }
                components.queryItems = queryItems
                guard let builtURL = components.url else {
                    throw URLError(.badURL)
                }
                request = URLRequest(url: builtURL)
                request.httpMethod = "GET"
                request.addBearer(accessToken)
            }
            request.addJSONHeaders()
            return request
        }
    }
}

private extension URLRequest {
    mutating func addBearer(_ token: String) {
        addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
    }

    mutating func addJSONHeaders() {
        addValue("application/json", forHTTPHeaderField: "Accept")
    }
}
