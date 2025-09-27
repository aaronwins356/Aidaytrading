import SwiftUI

struct TradesView: View {
    var body: some View {
        NavigationStack {
            List {
                Section("Recent trades") {
                    Text("Once connected, this list will show executed trades.")
                }
            }
            .navigationTitle("Trades")
        }
    }
}
