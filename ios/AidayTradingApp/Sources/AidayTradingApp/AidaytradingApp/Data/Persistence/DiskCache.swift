import Foundation

final class DiskCache {
    static let shared = DiskCache()

    private let fileManager: FileManager
    private let directoryURL: URL
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
        let baseURL = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first ?? URL(fileURLWithPath: NSTemporaryDirectory())
        self.directoryURL = baseURL.appendingPathComponent("Aidaytrading/Cache", isDirectory: true)
        self.encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .millisecondsSince1970
        self.decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .millisecondsSince1970
        createDirectoryIfNeeded()
    }

    func save<T: Encodable>(_ value: T, for key: String) throws {
        let url = directoryURL.appendingPathComponent(key)
        let data = try encoder.encode(value)
        try data.write(to: url, options: .atomic)
    }

    func load<T: Decodable>(_ type: T.Type, for key: String) -> T? {
        let url = directoryURL.appendingPathComponent(key)
        guard fileManager.fileExists(atPath: url.path) else { return nil }
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? decoder.decode(T.self, from: data)
    }

    func removeValue(for key: String) {
        let url = directoryURL.appendingPathComponent(key)
        try? fileManager.removeItem(at: url)
    }

    private func createDirectoryIfNeeded() {
        guard !fileManager.fileExists(atPath: directoryURL.path) else { return }
        try? fileManager.createDirectory(at: directoryURL, withIntermediateDirectories: true)
    }
}
