import Foundation

protocol APIClientProtocol {
    func send<T: Decodable>(_ request: APIRequestConvertible, decode type: T.Type) async throws -> T
    func send(_ request: APIRequestConvertible) async throws
}

final class APIClient: APIClientProtocol {
    private let urlSession: URLSession

    init(urlSession: URLSession = .shared) {
        self.urlSession = urlSession
    }

    func send<T: Decodable>(_ request: APIRequestConvertible, decode type: T.Type) async throws -> T {
        let request = try request.urlRequest
        try validateHTTPS(request)
        let (data, response) = try await urlSession.data(for: request)
        try HTTPURLResponse.validate(response: response, data: data)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(T.self, from: data)
    }

    func send(_ request: APIRequestConvertible) async throws {
        let request = try request.urlRequest
        try validateHTTPS(request)
        let (data, response) = try await urlSession.data(for: request)
        try HTTPURLResponse.validate(response: response, data: data)
        guard data.isEmpty == true else {
            return
        }
    }

    private func validateHTTPS(_ request: URLRequest) throws {
        guard request.url?.scheme?.lowercased() == "https" else {
            throw URLError(.appTransportSecurityRequiresSecureConnection)
        }
    }
}

private extension HTTPURLResponse {
    static func validate(response: URLResponse, data: Data) throws {
        guard let httpResponse = response as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }
        switch httpResponse.statusCode {
        case 200...299:
            return
        default:
            if let apiError = try? JSONDecoder().decode(APIErrorResponse.self, from: data) {
                throw apiError
            }
            throw URLError(.badServerResponse)
        }
    }
}
