import Charts
import SwiftUI

struct EquityChartView: View {
    let points: [EquityPoint]
    let delta: (amount: Decimal, percent: Decimal)?

    private var isPositive: Bool {
        guard let delta else { return true }
        return delta.amount >= 0
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionHeader(title: "Equity Curve")
            if points.isEmpty {
                EmptyStateView(title: "No data", message: "Equity history will appear once trades are closed.")
            } else {
                Chart(points) { point in
                    LineMark(
                        x: .value("Time", point.timestamp),
                        y: .value("Equity", point.equity as NSDecimalNumber)
                    )
                    .interpolationMethod(.catmullRom)
                    .foregroundStyle(isPositive ? Theme.accentGreen : Theme.accentRed)

                    AreaMark(
                        x: .value("Time", point.timestamp),
                        y: .value("Equity", point.equity as NSDecimalNumber)
                    )
                    .interpolationMethod(.catmullRom)
                    .foregroundStyle(
                        .linearGradient(
                            Gradient(colors: [
                                (isPositive ? Theme.accentGreen : Theme.accentRed).opacity(0.35),
                                Color.clear
                            ]),
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                }
                .chartXAxis {
                    AxisMarks(position: .bottom) { value in
                        AxisGridLine().foregroundStyle(Color.gridLine)
                        if let date = value.as(Date.self) {
                            AxisValueLabel(date.formatted(.dateTime.hour(.defaultDigits).minute()))
                                .foregroundStyle(Theme.secondaryText)
                        }
                    }
                }
                .chartYAxis {
                    AxisMarks(position: .leading) { value in
                        AxisGridLine().foregroundStyle(Color.gridLine)
                        if let decimal = value.as(NSDecimalNumber.self) {
                            let formatted = (decimal as Decimal).currencyString()
                            AxisValueLabel(formatted)
                                .foregroundStyle(Theme.secondaryText)
                        }
                    }
                }
                .frame(height: 240)
                .background(Theme.cardBackground)
                .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
                .accessibilityLabel("Equity chart, showing \(points.count) points")
            }
            if let delta {
                PnLChip(
                    amountText: delta.amount.currencyString(),
                    percentText: delta.percent.signedPercentString(),
                    isPositive: delta.amount >= 0
                )
            }
        }
    }
}
