import Foundation

struct AuthTokens: Codable {
    let accessToken: String
    let refreshToken: String
    let accessTokenExpiry: Date

    var isAccessTokenValid: Bool {
        accessTokenExpiry > Date()
    }
}

struct UserProfile: Codable, Identifiable {
    enum Role: String, Codable {
        case viewer
        case admin
    }

    let id: UUID
    let username: String
    let email: String
    let role: Role
    let approvalStatus: ApprovalStatus

    enum ApprovalStatus: String, Codable {
        case pending
        case approved
        case rejected
    }
}

struct UserSessionContext {
    let profile: UserProfile
    let tokens: AuthTokens
}

struct APIErrorResponse: Codable, Error {
    let message: String
}

// MARK: - Reporting Models

struct SystemStatus: Decodable {
    let running: Bool
    let uptimeSeconds: Int

    enum CodingKeys: String, CodingKey {
        case running
        case uptimeSeconds = "uptime_seconds"
    }
}

struct ProfitSummary: Decodable {
    let currentBalance: Decimal
    let totalPLAmount: Decimal
    let totalPLPercent: Decimal
    let winRate: Double

    enum CodingKeys: String, CodingKey {
        case currentBalance = "current_balance"
        case totalPLAmount = "total_pl_amount"
        case totalPLPercent = "total_pl_percent"
        case winRate = "win_rate"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        currentBalance = try container.decodeDecimal(forKey: .currentBalance)
        totalPLAmount = try container.decodeDecimal(forKey: .totalPLAmount)
        totalPLPercent = try container.decodeDecimal(forKey: .totalPLPercent)
        winRate = try container.decode(Double.self, forKey: .winRate)
    }

    init(currentBalance: Decimal, totalPLAmount: Decimal, totalPLPercent: Decimal, winRate: Double) {
        self.currentBalance = currentBalance
        self.totalPLAmount = totalPLAmount
        self.totalPLPercent = totalPLPercent
        self.winRate = winRate
    }
}

struct BalanceSnapshot: Decodable {
    let balance: Decimal

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        balance = try container.decodeDecimal(forKey: .balance)
    }

    private enum CodingKeys: String, CodingKey {
        case balance
    }
}

struct EquityCurvePoint: Decodable, Identifiable {
    let timestamp: Date
    let equity: Decimal

    var id: Date { timestamp }

    init(timestamp: Date, equity: Decimal) {
        self.timestamp = timestamp
        self.equity = equity
    }

    init(from decoder: Decoder) throws {
        var container = try decoder.unkeyedContainer()
        let timestampString = try container.decode(String.self)
        let equityString = try container.decode(String.self)
        guard let parsedTimestamp = ISO8601DateFormatter.apiFormatter.date(from: timestampString) else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Invalid timestamp: \(timestampString)")
        }
        guard let parsedEquity = Decimal(string: equityString) else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Invalid decimal: \(equityString)")
        }
        timestamp = parsedTimestamp
        equity = parsedEquity
    }
}

enum TradeSide: String, Decodable, CaseIterable, Identifiable {
    case buy
    case sell
    case short
    case cover

    var id: String { rawValue }
}

struct TradeRecord: Decodable, Identifiable {
    let id: Int
    let symbol: String
    let side: TradeSide
    let size: Decimal
    let pnl: Decimal
    let timestamp: Date

    enum CodingKeys: String, CodingKey {
        case id
        case symbol
        case side
        case size
        case pnl
        case timestamp
    }

    init(id: Int, symbol: String, side: TradeSide, size: Decimal, pnl: Decimal, timestamp: Date) {
        self.id = id
        self.symbol = symbol
        self.side = side
        self.size = size
        self.pnl = pnl
        self.timestamp = timestamp
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(Int.self, forKey: .id)
        symbol = try container.decode(String.self, forKey: .symbol)
        side = try container.decode(TradeSide.self, forKey: .side)
        size = try container.decodeDecimal(forKey: .size)
        pnl = try container.decodeDecimal(forKey: .pnl)
        let timestampString = try container.decode(String.self, forKey: .timestamp)
        guard let parsedTimestamp = ISO8601DateFormatter.apiFormatter.date(from: timestampString) else {
            throw DecodingError.dataCorruptedError(forKey: .timestamp, in: container, debugDescription: "Invalid timestamp")
        }
        timestamp = parsedTimestamp
    }
}

struct TradesPage: Decodable {
    let items: [TradeRecord]
    let page: Int
    let pageSize: Int
    let total: Int

    enum CodingKeys: String, CodingKey {
        case items
        case page
        case pageSize = "page_size"
        case total
    }
}

// MARK: - Decoding Helpers

extension KeyedDecodingContainer {
    func decodeDecimal(forKey key: Key) throws -> Decimal {
        if let stringValue = try? decode(String.self, forKey: key), let decimal = Decimal(string: stringValue) {
            return decimal
        }
        if let doubleValue = try? decode(Double.self, forKey: key) {
            return Decimal(doubleValue)
        }
        throw DecodingError.dataCorruptedError(forKey: key, in: self, debugDescription: "Unable to decode Decimal value")
    }
}
