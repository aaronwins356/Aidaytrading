import SwiftUI

struct PnLChip: View {
    let amountText: String
    let percentText: String
    let isPositive: Bool

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: isPositive ? "arrow.up.right" : "arrow.down.right")
                .font(.caption.bold())
            Text(amountText)
                .font(.caption.weight(.semibold))
            Text(percentText)
                .font(.caption)
                .foregroundColor(Theme.secondaryText)
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 12)
        .foregroundColor(isPositive ? Theme.accentGreen : Theme.accentRed)
        .background((isPositive ? Theme.accentGreen : Theme.accentRed).opacity(0.12))
        .clipShape(Capsule())
        .accessibilityLabel("Profit and loss: \(amountText), \(percentText)")
    }
}
