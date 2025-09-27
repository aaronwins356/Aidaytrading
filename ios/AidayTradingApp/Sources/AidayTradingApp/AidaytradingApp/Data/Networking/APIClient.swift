import Foundation

protocol AuthIntercepting {
    func authorize(_ request: URLRequest) async throws -> URLRequest
    func refreshAfterUnauthorized() async throws -> Bool
}

protocol APIClientPerforming {
    func get<T: Decodable>(_ endpoint: Endpoint) async throws -> T
}

final class APIClient: APIClientPerforming {
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
        self.decoder = APIClient.makeDecoder()
    }

    func get<T: Decodable>(_ endpoint: Endpoint) async throws -> T {
        var attempt = 0
        while attempt < 2 {
            do {
                var request = try endpoint.urlRequest(baseURL: baseURL)
                request = try await interceptor.authorize(request)
                let (data, response) = try await session.data(for: request)
                guard let httpResponse = response as? HTTPURLResponse else {
                    throw HTTPError.invalidResponse
                }
                if httpResponse.statusCode == 401, attempt == 0, try await interceptor.refreshAfterUnauthorized() {
                    attempt += 1
                    continue
                }
                guard (200...299).contains(httpResponse.statusCode) else {
                    throw HTTPError.statusCode(httpResponse.statusCode, data)
                }
                return try decoder.decode(T.self, from: data)
            } catch let error as DecodingError {
                throw HTTPError.decoding(error)
            } catch let error as HTTPError {
                throw error
            } catch {
                throw error
            }
        }
        throw HTTPError.invalidResponse
    }

    private static func makeDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .custom { decoder in
            let container = try decoder.singleValueContainer()
            if let milliseconds = try? container.decode(Double.self) {
                if milliseconds > 10_000_000_000 { // milliseconds
                    return Date(timeIntervalSince1970: milliseconds / 1000)
                } else {
                    return Date(timeIntervalSince1970: milliseconds)
                }
            }
            if let string = try? container.decode(String.self) {
                if let double = Double(string) {
                    if double > 10_000_000_000 {
                        return Date(timeIntervalSince1970: double / 1000)
                    } else {
                        return Date(timeIntervalSince1970: double)
                    }
                }
                if let date = ISO8601DateFormatter().date(from: string) {
                    return date
                }
            }
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: "Unsupported date format"))
        }
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        decoder.nonConformingFloatDecodingStrategy = .convertFromString(positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
        return decoder
    }
}
