import Foundation

struct StatusDTO: Decodable {
    let running: Bool
    let uptimeSeconds: Int

    func toEntity() -> Status {
        Status(running: running, uptimeSeconds: uptimeSeconds)
    }
}
