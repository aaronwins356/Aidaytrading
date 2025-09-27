import Foundation
@testable import AidayTradingApp

final class MockAPIClient: APIClientPerforming {
    var responses: [String: [Data]] = [:]

    func get<T>(_ endpoint: Endpoint) async throws -> T where T: Decodable {
        guard var queue = responses[endpoint.path], !queue.isEmpty else {
            throw URLError(.badURL)
        }
        let data = queue.removeFirst()
        responses[endpoint.path] = queue
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .millisecondsSince1970
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(T.self, from: data)
    }
}
