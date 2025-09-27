import SwiftUI

struct NotificationBannerView: View {
    let notification: AppNotification

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Text(icon)
                .font(.title2)
                .padding(8)
                .background(.thinMaterial)
                .clipShape(Circle())
            VStack(alignment: .leading, spacing: 4) {
                Text(notification.title)
                    .font(.headline)
                    .foregroundStyle(.white)
                Text(notification.body)
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.8))
            }
            Spacer()
        }
        .padding()
        .background(.ultraThinMaterial.opacity(0.9))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .shadow(color: .black.opacity(0.2), radius: 12, x: 0, y: 6)
        .padding(.horizontal)
        .accessibilityElement(children: .combine)
    }

    private var icon: String {
        switch notification.kind {
        case .botEvent:
            if let running = notification.payloadDictionary?["running"] as? Bool {
                return running ? "🚀" : "✋"
            }
            return "🚀"
        case .report:
            return "📊"
        case .system:
            return "🛰️"
        }
    }
}
