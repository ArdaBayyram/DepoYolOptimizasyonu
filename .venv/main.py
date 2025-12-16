import sys
import random
import heapq
import math
import time
import itertools
import numpy as np
from collections import deque
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QMessageBox,
                             QSpinBox, QGroupBox, QCheckBox, QGridLayout)
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# --- AYARLAR ---
GRID_SIZE = 80
OBSTACLE_DENSITY = 0.8


# --------------------------
# ALGORİTMA MOTORU
# --------------------------
class Pathfinder:
    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def heuristic(self, a, b, use_heuristic):
        return (abs(a[0] - b[0]) + abs(a[1] - b[1])) if use_heuristic else 0

    def get_neighbors(self, node):
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = node[0] + dx, node[1] + dy
            if 0 <= nx < self.rows and 0 <= ny < self.cols:
                if self.grid[nx][ny] == 0:
                    neighbors.append((nx, ny))
        return neighbors

    def find_path_weighted(self, start, goal, use_heuristic=True):
        open_set = []
        heapq.heappush(open_set, (0, 0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal, use_heuristic)}
        visited_nodes = set()
        counter = 1

        while open_set:
            current = heapq.heappop(open_set)[2]
            visited_nodes.add(current)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1], len(visited_nodes)

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    h = self.heuristic(neighbor, goal, use_heuristic)
                    f_score[neighbor] = tentative_g_score + h
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
                    counter += 1
        return [], len(visited_nodes)

    def find_path_bfs(self, start, goal):
        queue = deque([start])
        came_from = {start: None}
        visited_nodes = set([start])

        while queue:
            current = queue.popleft()
            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                return path[::-1], len(visited_nodes)

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    came_from[neighbor] = current
                    queue.append(neighbor)
        return [], len(visited_nodes)


# --------------------------
# SİMÜLASYON YÖNETİCİSİ
# --------------------------
class WarehouseSimulation(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE))
        self.items = []
        self.start_pos = (0, 0)
        self.active_paths = []
        self.pathfinder = None
        self.generate_warehouse()

    def generate_warehouse(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE))
        c = 1
        while c < GRID_SIZE - 1:
            if abs(c - GRID_SIZE // 2) < 2:
                c += 1
                continue
            if random.random() < 0.7:
                shelf_width = 1
                if c < GRID_SIZE - 2 and random.random() < 0.3: shelf_width = 2
                r = 1
                while r < GRID_SIZE - 1:
                    block_len = random.randint(3, 8)
                    if r + block_len >= GRID_SIZE - 1: block_len = GRID_SIZE - 1 - r
                    for k in range(block_len):
                        self.grid[r + k][c] = 1
                        if shelf_width == 2: self.grid[r + k][c + 1] = 1
                    r += block_len
                    gap_len = random.randint(1, 3)
                    r += gap_len
                c += shelf_width + 1
            else:
                c += 1

        mid_col = GRID_SIZE // 2
        safe_r = 0
        while safe_r < GRID_SIZE and self.grid[safe_r][mid_col] == 1:
            safe_r += 1
        if safe_r >= GRID_SIZE: safe_r = 0
        self.start_pos = (safe_r, mid_col)

        self.pathfinder = Pathfinder(self.grid)
        self.items = []
        self.active_paths = []
        self.update()

    def generate_order(self, count):
        self.items = []
        attempts = 0
        while len(self.items) < count and attempts < 3000:
            rx, ry = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
            if self.grid[rx][ry] == 0 and (rx, ry) != self.start_pos:
                if (rx, ry) not in self.items:
                    self.items.append((rx, ry))
            attempts += 1
        self.active_paths = []
        self.update()

    def calculate_metrics(self, order, algo_type='astar'):
        total_dist = 0;
        total_visited = 0;
        total_turns = 0
        full_path = []
        curr = self.start_pos
        last_dx, last_dy = 0, 0

        t_start = time.time()

        for item in order:
            if algo_type == 'astar':
                path, visited = self.pathfinder.find_path_weighted(curr, item, use_heuristic=True)
            elif algo_type == 'dijkstra':
                path, visited = self.pathfinder.find_path_weighted(curr, item, use_heuristic=False)
            elif algo_type == 'bfs':
                path, visited = self.pathfinder.find_path_bfs(curr, item)
            else:
                path, visited = [], 0

            total_visited += visited
            if len(path) > 1:
                total_dist += len(path) - 1
                for i in range(len(path) - 1):
                    p1, p2 = path[i], path[i + 1]
                    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                    if (last_dx, last_dy) != (0, 0) and (dx, dy) != (last_dx, last_dy):
                        total_turns += 1
                    last_dx, last_dy = dx, dy

                if not full_path:
                    full_path.extend(path)
                else:
                    full_path.extend(path[1:])
            curr = item

        t_end = time.time()
        return {
            'dist': total_dist,
            'vis': total_visited,
            'turns': total_turns,
            'time': (t_end - t_start) * 1000,
            'path': full_path
        }

    def run_full_analysis(self):
        if not self.items: return None
        results = {'KNN': {}, 'SA': {}, 'RND': {}, 'BF': None}

        random_state = random.getstate()
        random.seed(str(self.items))

        # --- 1. K-NN ROTA ---
        unvisited = sorted(self.items.copy())
        curr = self.start_pos;
        knn_order = []
        while unvisited:
            nearest = min(unvisited, key=lambda x: abs(x[0] - curr[0]) + abs(x[1] - curr[1]))
            knn_order.append(nearest);
            unvisited.remove(nearest);
            curr = nearest

        results['KNN']['A*'] = self.calculate_metrics(knn_order, 'astar')
        results['KNN']['Dijkstra'] = self.calculate_metrics(knn_order, 'dijkstra')
        results['KNN']['BFS'] = self.calculate_metrics(knn_order, 'bfs')

        # --- 2. SA ROTA ---
        curr_sol = sorted(self.items.copy());
        random.shuffle(curr_sol)

        def cost(sol):
            d = 0;
            c = self.start_pos
            for i in sol: d += abs(i[0] - c[0]) + abs(i[1] - c[1]); c = i
            return d

        best_sol = list(curr_sol);
        best_cost = cost(curr_sol)
        temp = 1000.0
        for _ in range(1000):
            new_sol = list(curr_sol)
            i1, i2 = random.sample(range(len(new_sol)), 2)
            new_sol[i1], new_sol[i2] = new_sol[i2], new_sol[i1]
            current_c = cost(curr_sol);
            new_c = cost(new_sol)
            if new_c < current_c or random.random() < math.exp((current_c - new_c) / temp):
                curr_sol = new_sol
                if cost(curr_sol) < best_cost: best_sol = list(curr_sol); best_cost = cost(curr_sol)
            temp *= 0.99

        results['SA']['A*'] = self.calculate_metrics(best_sol, 'astar')
        results['SA']['Dijkstra'] = self.calculate_metrics(best_sol, 'dijkstra')
        results['SA']['BFS'] = self.calculate_metrics(best_sol, 'bfs')

        # --- 3. BRUTE FORCE ---
        if len(self.items) <= 11:
            best_bf_order = None
            min_bf_dist = float('inf')
            for p in itertools.permutations(self.items):
                d = 0;
                c = self.start_pos
                for i in p: d += abs(i[0] - c[0]) + abs(i[1] - c[1]); c = i
                if d < min_bf_dist: min_bf_dist = d; best_bf_order = list(p)
            results['BF'] = self.calculate_metrics(best_bf_order, 'astar')
        else:
            results['BF'] = None

        # --- 4. RASTGELE ROTA ---
        rnd_items = sorted(self.items.copy());
        random.shuffle(rnd_items)
        results['RND']['A*'] = self.calculate_metrics(rnd_items, 'astar')

        random.setstate(random_state)
        return results

    def clear_paths(self):
        self.active_paths = []
        self.update()

    def add_path(self, path, color, width, style=Qt.SolidLine):
        self.active_paths.append({'path': path, 'color': color, 'width': width, 'style': style})
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        w = self.width() / GRID_SIZE
        h = self.height() / GRID_SIZE

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if self.grid[r][c] == 1:
                    painter.setBrush(QBrush(QColor(60, 60, 60)))
                else:
                    painter.setBrush(QBrush(QColor(245, 245, 245)))
                painter.setPen(QPen(QColor(220, 220, 220)))
                painter.drawRect(int(c * w), int(r * h), int(w), int(h))

        for path_data in self.active_paths:
            path = path_data['path']
            if not path: continue
            pen = QPen(path_data['color'], path_data['width'])
            pen.setStyle(path_data.get('style', Qt.SolidLine))
            pen.setJoinStyle(Qt.MiterJoin)
            painter.setPen(pen)
            for i in range(len(path) - 1):
                p1 = path[i];
                p2 = path[i + 1]
                painter.drawLine(int(p1[1] * w + w / 2), int(p1[0] * h + h / 2),
                                 int(p2[1] * w + w / 2), int(p2[0] * h + h / 2))

        painter.setBrush(QBrush(QColor(0, 180, 0)));
        painter.setPen(Qt.NoPen)
        for i in self.items:
            painter.drawEllipse(int(i[1] * w + w * 0.2), int(i[0] * h + h * 0.2), int(w * 0.6), int(h * 0.6))

        painter.setBrush(QBrush(QColor(0, 0, 200)))
        painter.drawEllipse(int(self.start_pos[1] * w + w * 0.2), int(self.start_pos[0] * h + h * 0.2), int(w * 0.6),
                            int(h * 0.6))


# --------------------------
# ANA ARAYÜZ
# --------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Algoritma Laboratuvarı: K-NN, SA ve BF Analizi")
        self.setGeometry(50, 50, 1500, 900)
        self.results_cache = None

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        left_layout = QVBoxLayout()
        self.sim = WarehouseSimulation()
        left_layout.addWidget(self.sim, stretch=6)

        ctrl = QGroupBox("Kontroller")
        clayout = QVBoxLayout()
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Sipariş Sayısı (Max 100):"))
        self.spin = QSpinBox();
        self.spin.setValue(6);
        self.spin.setRange(2, 100)
        h1.addWidget(self.spin)
        clayout.addLayout(h1)
        h2 = QHBoxLayout()
        b1 = QPushButton("Depo Üret");
        b1.clicked.connect(self.reset)
        b2 = QPushButton("Sipariş Ver");
        b2.clicked.connect(self.order)
        b3 = QPushButton("HESAPLA");
        b3.clicked.connect(self.run_calculations)
        b3.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        h2.addWidget(b1);
        h2.addWidget(b2);
        h2.addWidget(b3)
        clayout.addLayout(h2)
        ctrl.setLayout(clayout)
        left_layout.addWidget(ctrl, stretch=1)

        vis_group = QGroupBox("Görselleştirme")
        vis_layout = QGridLayout()
        self.check_knn_astar = QCheckBox("K-NN + A* (Mavi)");
        self.check_knn_astar.setChecked(True)
        self.check_knn_dijk = QCheckBox("K-NN + Dij (Turkuaz)")
        self.check_knn_bfs = QCheckBox("K-NN + BFS (Mor)")
        self.check_sa_astar = QCheckBox("SA + A* (Yeşil)");
        self.check_sa_astar.setChecked(True)
        self.check_sa_dijk = QCheckBox("SA + Dij (Açık Yeşil)")
        self.check_sa_bfs = QCheckBox("SA + BFS (Sarı)")
        self.check_bf = QCheckBox("BRUTE FORCE (Altın)")

        for chk in [self.check_knn_astar, self.check_knn_dijk, self.check_knn_bfs,
                    self.check_sa_astar, self.check_sa_dijk, self.check_sa_bfs, self.check_bf]:
            chk.stateChanged.connect(self.update_visualization)

        vis_layout.addWidget(self.check_knn_astar, 0, 0);
        vis_layout.addWidget(self.check_knn_dijk, 1, 0);
        vis_layout.addWidget(self.check_knn_bfs, 2, 0)
        vis_layout.addWidget(self.check_sa_astar, 0, 1);
        vis_layout.addWidget(self.check_sa_dijk, 1, 1);
        vis_layout.addWidget(self.check_sa_bfs, 2, 1)
        vis_layout.addWidget(self.check_bf, 3, 0, 1, 2)
        vis_group.setLayout(vis_layout)
        left_layout.addWidget(vis_group, stretch=1)

        right_layout = QVBoxLayout()
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)
        layout.addLayout(left_layout, stretch=4)
        layout.addLayout(right_layout, stretch=5)

    def reset(self):
        self.sim.generate_warehouse()
        self.figure.clear();
        self.canvas.draw()
        self.results_cache = None

    def order(self):
        self.sim.generate_order(self.spin.value())
        self.results_cache = None

    def run_calculations(self):
        if self.spin.value() > 11:
            QMessageBox.warning(self, "Uyarı", "Ürün sayısı 9'dan fazla! Brute Force hesaplanmayacak.")
        self.results_cache = self.sim.run_full_analysis()
        if not self.results_cache: return
        self.update_visualization()
        self.draw_graphs()

    def update_visualization(self):
        if not self.results_cache: return
        self.sim.clear_paths()
        res = self.results_cache

        if self.check_knn_astar.isChecked(): self.sim.add_path(res['KNN']['A*']['path'], QColor(0, 0, 255, 120), 6)
        if self.check_knn_dijk.isChecked(): self.sim.add_path(res['KNN']['Dijkstra']['path'], QColor(0, 255, 255, 120),
                                                              4, Qt.DashLine)
        if self.check_knn_bfs.isChecked(): self.sim.add_path(res['KNN']['BFS']['path'], QColor(128, 0, 128, 120), 2,
                                                             Qt.DotLine)

        if self.check_sa_astar.isChecked(): self.sim.add_path(res['SA']['A*']['path'], QColor(0, 255, 0, 200), 3)
        if self.check_sa_dijk.isChecked(): self.sim.add_path(res['SA']['Dijkstra']['path'], QColor(144, 238, 144, 200),
                                                             2, Qt.DashLine)
        if self.check_sa_bfs.isChecked(): self.sim.add_path(res['SA']['BFS']['path'], QColor(255, 215, 0, 200), 2,
                                                            Qt.DotLine)

        if self.check_bf.isChecked() and res['BF']:
            self.sim.add_path(res['BF']['path'], QColor(255, 215, 0, 255), 4)

    def draw_graphs(self):
        self.figure.clear()
        res = self.results_cache

        # Grafikleri 2x3 grid olarak ayarla (toplam 6 grafik)
        ax1 = self.figure.add_subplot(231)
        ax2 = self.figure.add_subplot(232)
        ax3 = self.figure.add_subplot(233)
        ax4 = self.figure.add_subplot(234)
        ax5 = self.figure.add_subplot(235)
        ax6 = self.figure.add_subplot(236)  # VERİMLİLİK PUANI

        # Grafik Verileri (RND YOK)
        algos = ['K-NN', 'SA']
        dists = [res['KNN']['A*']['dist'], res['SA']['A*']['dist']]
        colors = ['blue', 'green']

        if res['BF']:
            algos.append('BF (Opt)')
            dists.append(res['BF']['dist'])
            colors.append('gold')

        # 1. Genel Rota Mesafesi (BF Dahil)
        ax1.bar(algos, dists, color=colors)
        ax1.set_title('EN Kısa Mesafe')
        ax1.set_ylabel('Birim')
        for i, v in enumerate(dists): ax1.text(i, v, str(v), ha='center', va='bottom', fontsize=8, fontweight='bold')

        # 2. Algoritma Tutarlılığı (Sağlama)
        labels = ['A*', 'Dij', 'BFS']
        knn_lens = [res['KNN']['A*']['dist'], res['KNN']['Dijkstra']['dist'], res['KNN']['BFS']['dist']]
        sa_lens = [res['SA']['A*']['dist'], res['SA']['Dijkstra']['dist'], res['SA']['BFS']['dist']]
        x = np.arange(len(labels));
        width = 0.35
        ax2.bar(x - width / 2, knn_lens, width, label='K-NN', color='blue', alpha=0.6)
        ax2.bar(x + width / 2, sa_lens, width, label='SA', color='green', alpha=0.6)
        ax2.set_title('Mesafe')
        ax2.set_xticks(x);
        ax2.set_xticklabels(labels)
        ax2.legend(fontsize='x-small')

        # 3. Verimlilik
        knn_vis = [res['KNN']['A*']['vis'], res['KNN']['Dijkstra']['vis'], res['KNN']['BFS']['vis']]
        sa_vis = [res['SA']['A*']['vis'], res['SA']['Dijkstra']['vis'], res['SA']['BFS']['vis']]
        ax3.bar(x - width / 2, knn_vis, width, color='blue', alpha=0.6)
        ax3.bar(x + width / 2, sa_vis, width, color='green', alpha=0.6)
        ax3.set_title('Verimlilik (Gezilen Düğüm)')
        ax3.set_xticks(x);
        ax3.set_xticklabels(labels)

        # 4. Süre
        knn_time = [res['KNN']['A*']['time'], res['KNN']['Dijkstra']['time'], res['KNN']['BFS']['time']]
        sa_time = [res['SA']['A*']['time'], res['SA']['Dijkstra']['time'], res['SA']['BFS']['time']]
        ax4.bar(x - width / 2, knn_time, width, color='blue', alpha=0.6)
        ax4.bar(x + width / 2, sa_time, width, color='green', alpha=0.6)
        ax4.set_title('Hesaplama Süresi (ms)')
        ax4.set_xticks(x);
        ax4.set_xticklabels(labels)

        # 5. Optimizasyon Kazancı (BF Dahil)
        rnd_dist = res['RND']['A*']['dist']
        gains = []
        glabels = []
        gcolors = []

        if rnd_dist > 0:
            gains.append((rnd_dist - res['KNN']['A*']['dist']) / rnd_dist * 100)
            glabels.append('K-NN')
            gcolors.append('blue')

            gains.append((rnd_dist - res['SA']['A*']['dist']) / rnd_dist * 100)
            glabels.append('SA')
            gcolors.append('green')

            if res['BF']:
                gains.append((rnd_dist - res['BF']['dist']) / rnd_dist * 100)
                glabels.append('BF')
                gcolors.append('gold')

        bars = ax5.bar(glabels, gains, color=gcolors)
        ax5.set_title('Optimizasyon Kazancı (%)')
        ax5.set_ylim(0, 100)
        for b in bars: ax5.text(b.get_x() + b.get_width() / 2, b.get_height(), f'%{b.get_height():.1f}', ha='center',
                                va='bottom')

        # ---------------------------------------------------------
        # 6. (DÜZELTİLMİŞ) VERİMLİLİK PUANI
        # Mantık: BF'nin süresi artsa bile puanı DÜŞMELİ.
        # Eski formüldeki +10 sabiti kaldırıldı, zaman etkisi artırıldı.
        # ---------------------------------------------------------

        comp_labels = []
        scores = []
        bar_colors = []

        # Puan hesaplama yardımcı fonksiyonu (YENİ FORMÜL)
        def calculate_score(dist, time_ms):
            if dist == 0: return 0
            # Buffer'ı +10'dan +0.5'e indirdim.
            # Artık 0.1ms (KNN) ile 20ms (BF) arasında 200 kat puan farkı oluşacak.
            return 10000000 / (dist * (time_ms + 0.5))

        # K-NN varyasyonları
        for sub in ['A*', 'Dijkstra', 'BFS']:
            comp_labels.append(f"KNN\n{sub}")
            s = calculate_score(res['KNN'][sub]['dist'], res['KNN'][sub]['time'])
            scores.append(s)
            bar_colors.append('lightblue')

        # SA varyasyonları
        for sub in ['A*', 'Dijkstra', 'BFS']:
            comp_labels.append(f"SA\n{sub}")
            s = calculate_score(res['SA'][sub]['dist'], res['SA'][sub]['time'])
            scores.append(s)
            bar_colors.append('lightgreen')

        # Brute Force
        if res['BF']:
            comp_labels.append("BF")
            s = calculate_score(res['BF']['dist'], res['BF']['time'])
            scores.append(s)
            bar_colors.append('gold')

        # Çizim
        score_bars = ax6.bar(comp_labels, scores, color=bar_colors, edgecolor='black', alpha=0.8)
        ax6.set_title('TOPLAM VERİMLİLİK PUANI\n(Hız Kriteri Artırıldı)', fontsize=9, fontweight='bold',
                      color='darkred')
        ax6.set_ylabel('Verimlilik Skoru', fontsize=9)
        ax6.tick_params(axis='x', labelsize=7, rotation=15)
        ax6.grid(axis='y', linestyle='--', alpha=0.3)

        # Değerleri yaz
        for bar in score_bars:
            h = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width() / 2, h, f"{int(h)}", ha='center', va='bottom', fontsize=8,
                     fontweight='bold')

        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())