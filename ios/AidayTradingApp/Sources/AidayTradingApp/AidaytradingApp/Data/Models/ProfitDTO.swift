import Foundation

struct ProfitDTO: Decodable {
    let balanceUsd: Decimal
    let pnlAbs: Decimal
    let pnlPct: Decimal
    let winRate: Double

    private enum CodingKeys: String, CodingKey {
        case balanceUsd = "balance_usd"
        case pnlAbs = "pnl_abs"
        case pnlPct = "pnl_pct"
        case winRate = "win_rate"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        balanceUsd = try container.decode(Decimal.self, forKey: .balanceUsd)
        pnlAbs = try container.decode(Decimal.self, forKey: .pnlAbs)
        pnlPct = try container.decode(Decimal.self, forKey: .pnlPct)
        winRate = try container.decode(Double.self, forKey: .winRate)
    }

    func toEntity() -> ProfitSnapshot {
        ProfitSnapshot(balance: balanceUsd, pnlAbsolute: pnlAbs, pnlPercent: pnlPct, winRate: winRate)
    }
}
