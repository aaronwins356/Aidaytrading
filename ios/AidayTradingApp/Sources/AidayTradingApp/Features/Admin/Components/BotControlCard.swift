import SwiftUI

struct BotControlCard: View {
    let status: BotStatus?
    let isBusy: Bool
    let onStart: () -> Void
    let onStop: () -> Void
    let onModeChange: (BotMode) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Bot Control")
                        .font(.headline)
                        .foregroundColor(Theme.primaryText)
                    Text(statusDescription)
                        .font(.subheadline)
                        .foregroundColor(statusColor)
                }
                Spacer()
                if let status {
                    Text(status.mode.description.uppercased())
                        .font(.caption.bold())
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(status.mode.isPaperTrading ? Color.blue.opacity(0.2) : Theme.accentGreen.opacity(0.2))
                        .foregroundColor(status.mode.isPaperTrading ? Color.blue : Theme.accentGreen)
                        .clipShape(Capsule())
                }
            }

            HStack(spacing: 12) {
                Button(action: onStart) {
                    Label("Start", systemImage: "play.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(AdminFilledButtonStyle(color: Theme.accentGreen))
                .disabled(isBusy || (status?.running ?? false))

                Button(action: onStop) {
                    Label("Stop", systemImage: "stop.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(AdminFilledButtonStyle(color: Theme.accentRed))
                .disabled(isBusy || !(status?.running ?? false))
            }

            Picker("Mode", selection: Binding(
                get: { status?.mode ?? .paper },
                set: { newValue in
                    if status?.mode != newValue {
                        onModeChange(newValue)
                    }
                }
            )) {
                Text("Paper").tag(BotMode.paper)
                Text("Live").tag(BotMode.live)
            }
            .pickerStyle(.segmented)
            .disabled(isBusy || status == nil)
        }
        .padding()
        .background(Theme.cardBackground)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .shadow(color: Color.black.opacity(0.2), radius: 8, x: 0, y: 4)
    }

    private var statusDescription: String {
        guard let status else { return "Loading status..." }
        return status.running ? "Bot is Running" : "Bot is Stopped"
    }

    private var statusColor: Color {
        guard let status else { return Theme.secondaryText }
        return status.running ? Theme.accentGreen : Theme.accentRed
    }
}

struct AdminFilledButtonStyle: ButtonStyle {
    let color: Color

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.headline)
            .padding()
            .background(color.opacity(configuration.isPressed ? 0.7 : 1.0))
            .foregroundColor(.white)
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}
