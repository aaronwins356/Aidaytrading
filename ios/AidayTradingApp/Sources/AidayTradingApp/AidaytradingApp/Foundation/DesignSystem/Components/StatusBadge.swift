import SwiftUI

struct StatusBadge: View {
    let isRunning: Bool
    let uptime: String?

    var body: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(isRunning ? Theme.accentGreen : Theme.accentRed)
                .frame(width: 10, height: 10)
                .accessibilityHidden(true)
            Text(isRunning ? "Running" : "Stopped")
                .font(.subheadline.weight(.semibold))
                .foregroundColor(isRunning ? Theme.accentGreen : Theme.accentRed)
            if let uptime {
                Text(uptime)
                    .font(.caption)
                    .foregroundColor(Theme.secondaryText)
            }
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 12)
        .background(Theme.cardBackground)
        .clipShape(Capsule())
        .accessibilityLabel(isRunning ? "Bot running" : "Bot stopped")
    }
}
