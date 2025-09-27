import Foundation

enum HTTPError: Error, LocalizedError {
    case nonHTTPS
    case invalidResponse
    case statusCode(Int, Data?)
    case decoding(Error)

    var errorDescription: String? {
        switch self {
        case .nonHTTPS:
            return "Insecure connection attempted."
        case .invalidResponse:
            return "The server returned an invalid response."
        case let .statusCode(code, _):
            return "Server responded with status code \(code)."
        case let .decoding(error):
            return "Failed to decode response: \(error.localizedDescription)"
        }
    }
}
