import Foundation

enum APIEnvironment {
    static var baseURL: URL { AppConfig.baseURL }
}

protocol APIRequestConvertible {
    var urlRequest: URLRequest { get throws }
}
