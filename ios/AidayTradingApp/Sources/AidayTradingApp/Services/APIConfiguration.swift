import Foundation

enum APIEnvironment {
    static let baseURL = URL(string: "https://api.aidaytrading.com")!
}

protocol APIRequestConvertible {
    var urlRequest: URLRequest { get throws }
}
