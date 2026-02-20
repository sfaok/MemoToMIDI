//
//  ContentView.swift
//  MemoToMIDI
//
//  Created by JAMES ROBERTSON on 2026-02-19.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        NavigationStack {
            RecordingView()
                .navigationTitle("MemoToMIDI")
        }
    }
}

#Preview {
    ContentView()
}
