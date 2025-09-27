import SwiftUI

struct ErrorStateView: View {
    let title: String
    let message: String
    let retryTitle: String
    let retry: () -> Void

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 32))
                .foregroundColor(Theme.accentRed)
            Text(title)
                .font(.headline)
                .foregroundColor(Theme.primaryText)
            Text(message)
                .font(.subheadline)
                .multilineTextAlignment(.center)
                .foregroundColor(Theme.secondaryText)
            Button(retryTitle, action: retry)
                .font(.subheadline.bold())
                .padding(.vertical, 8)
                .padding(.horizontal, 16)
                .background(Theme.accentGreen)
                .foregroundColor(.black)
                .clipShape(Capsule())
        }
        .padding(24)
        .frame(maxWidth: .infinity)
        .background(Theme.cardBackground)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .accessibilityElement(children: .combine)
    }
}
