import Foundation

struct AdminUser: Identifiable, Equatable, Codable {
    enum Role: String, Codable, CaseIterable, Identifiable {
        case viewer
        case admin

        var id: String { rawValue }

        var displayName: String {
            switch self {
            case .viewer: return "Viewer"
            case .admin: return "Admin"
            }
        }
    }

    enum Status: String, Codable, CaseIterable, Identifiable {
        case active
        case disabled
        case pending

        var id: String { rawValue }

        var displayName: String {
            switch self {
            case .active: return "Active"
            case .disabled: return "Disabled"
            case .pending: return "Pending"
            }
        }

        var tintColor: String {
            switch self {
            case .active: return "green"
            case .disabled: return "red"
            case .pending: return "orange"
            }
        }

        var iconName: String {
            switch self {
            case .active: return "checkmark.seal"
            case .disabled: return "xmark.octagon"
            case .pending: return "clock"
            }
        }
    }

    let id: UUID
    let username: String
    let email: String
    var role: Role
    var status: Status

    var isPending: Bool { status == .pending }
}

struct RiskConfiguration: Equatable, Codable {
    var maxDrawdownPercent: Double
    var dailyLossLimitPercent: Double
    var riskPerTrade: Double
    var maxOpenPositions: Int
    var atrStopLossMultiplier: Double
    var atrTakeProfitMultiplier: Double

    static let ranges = RiskParameterRanges()

    mutating func clamp() {
        maxDrawdownPercent = Self.ranges.maxDrawdownPercent.clamped(maxDrawdownPercent)
        dailyLossLimitPercent = Self.ranges.dailyLossLimitPercent.clamped(dailyLossLimitPercent)
        riskPerTrade = Self.ranges.riskPerTrade.clamped(riskPerTrade)
        maxOpenPositions = Int(Self.ranges.maxOpenPositions.clamped(Double(maxOpenPositions)))
        atrStopLossMultiplier = Self.ranges.atrStopLossMultiplier.clamped(atrStopLossMultiplier)
        atrTakeProfitMultiplier = Self.ranges.atrTakeProfitMultiplier.clamped(atrTakeProfitMultiplier)
    }

    func clamped() -> RiskConfiguration {
        var copy = self
        copy.clamp()
        return copy
    }

    func validate() -> Bool {
        self == clamped()
    }

    static let `default` = RiskConfiguration(
        maxDrawdownPercent: 20,
        dailyLossLimitPercent: 5,
        riskPerTrade: 0.02,
        maxOpenPositions: 3,
        atrStopLossMultiplier: 1.5,
        atrTakeProfitMultiplier: 2.5
    )
}

struct RiskParameterRanges {
    let maxDrawdownPercent = 5.0...90.0
    let dailyLossLimitPercent = 1.0...50.0
    let riskPerTrade = 0.005...0.1
    let maxOpenPositions = 1.0...10.0
    let atrStopLossMultiplier = 0.5...3.0
    let atrTakeProfitMultiplier = 1.0...5.0
}

enum BotMode: String, Codable {
    case paper
    case live

    init(paperTrading: Bool) {
        self = paperTrading ? .paper : .live
    }

    var isPaperTrading: Bool { self == .paper }

    var description: String {
        switch self {
        case .paper: return "Paper"
        case .live: return "Live"
        }
    }
}

struct BotStatus: Equatable, Codable {
    var running: Bool
    var mode: BotMode
    var lastUpdated: Date

    var statusDescription: String {
        running ? "Running" : "Stopped"
    }
}

struct AdminChangeLogEntry: Identifiable, Equatable, Codable {
    enum Category: String, Codable {
        case risk
        case user
        case bot
    }

    let id: UUID
    let timestamp: Date
    let actor: String
    let summary: String
    let details: String
    let category: Category
}

private extension ClosedRange where Bound == Double {
    func clamped(_ value: Double) -> Double {
        if value < lowerBound { return lowerBound }
        if value > upperBound { return upperBound }
        return value
    }
}
