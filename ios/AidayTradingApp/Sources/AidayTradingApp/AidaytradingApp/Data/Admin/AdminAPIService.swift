import Foundation

protocol AdminAPIServiceProtocol {
    func fetch<T: Decodable>(_ endpoint: AdminEndpoint, as type: T.Type) async throws -> T
    func send(_ endpoint: AdminEndpoint) async throws
}

actor AdminAPIService: AdminAPIServiceProtocol {
    private let session: URLSession
    private let baseURL: URL
    private let interceptor: AuthIntercepting
    private let decoder: JSONDecoder

    init(
        session: URLSession = .shared,
        baseURL: URL = AppConfig.baseURL,
        interceptor: AuthIntercepting = AuthInterceptor()
    ) {
        self.session = session
        self.baseURL = baseURL
        self.interceptor = interceptor
        self.decoder = AdminAPIService.makeDecoder()
    }

    func fetch<T: Decodable>(_ endpoint: AdminEndpoint, as type: T.Type) async throws -> T {
        let request = try await authorizedRequest(for: endpoint)
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try decoder.decode(T.self, from: data)
    }

    func send(_ endpoint: AdminEndpoint) async throws {
        let request = try await authorizedRequest(for: endpoint)
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        _ = data
    }

    private func authorizedRequest(for endpoint: AdminEndpoint) async throws -> URLRequest {
        var request = try endpoint.urlRequest(baseURL: baseURL)
        request = try await interceptor.authorize(request)
        return request
    }

    private func validate(response: URLResponse, data: Data) throws {
        guard let httpResponse = response as? HTTPURLResponse else {
            throw HTTPError.invalidResponse
        }
        guard (200...299).contains(httpResponse.statusCode) else {
            throw HTTPError.statusCode(httpResponse.statusCode, data)
        }
    }

    private static func makeDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }
}
