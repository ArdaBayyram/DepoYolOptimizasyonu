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
GRID_SIZE = 55
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

    # Calculate Metrics: Sadece yürütülen yolun çizimi ve metrikleri
    def calculate_metrics(self, order, algo_type='astar'):
        total_dist = 0;
        total_visited = 0;
        total_turns = 0
        full_path = []
        curr = self.start_pos
        last_dx, last_dy = 0, 0

        t_start = time.time()  # Sadece path bulma süresi

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
        # 'time' değişkeni burada sadece yürüme süresini tutuyor.
        # Run_full_analysis içinde buna "Düşünme Süresi" eklenecek.
        return {
            'dist': total_dist,
            'vis': total_visited,
            'turns': total_turns,
            'time': (t_end - t_start) * 1000,
            'path': full_path
        }

    # --- HESAPLAMA SÜRELERİNİ ÖLÇEN FONKSİYON ---
    def run_full_analysis(self):
        if not self.items: return None
        results = {'KNN': {}, 'SA': {}, 'RND': {}, 'BF': None}

        random_state = random.getstate()
        random.seed(str(self.items))

        # --- 1. K-NN ROTA (Düşünme Süresi) ---
        t_knn_start = time.time()

        unvisited = sorted(self.items.copy())
        curr = self.start_pos;
        knn_order = []
        while unvisited:
            nearest = min(unvisited, key=lambda x: abs(x[0] - curr[0]) + abs(x[1] - curr[1]))
            knn_order.append(nearest);
            unvisited.remove(nearest);
            curr = nearest

        t_knn_end = time.time()
        knn_comp_time = (t_knn_end - t_knn_start) * 1000

        # Sonuçlara ekle
        r_astar = self.calculate_metrics(knn_order, 'astar')
        r_astar['time'] += knn_comp_time
        results['KNN']['A*'] = r_astar

        r_dijk = self.calculate_metrics(knn_order, 'dijkstra')
        r_dijk['time'] += knn_comp_time
        results['KNN']['Dijkstra'] = r_dijk

        r_bfs = self.calculate_metrics(knn_order, 'bfs')
        r_bfs['time'] += knn_comp_time
        results['KNN']['BFS'] = r_bfs

        # --- 2. SA ROTA (Düşünme Süresi) ---
        t_sa_start = time.time()

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

        t_sa_end = time.time()
        sa_comp_time = (t_sa_end - t_sa_start) * 1000

        r_sa_astar = self.calculate_metrics(best_sol, 'astar')
        r_sa_astar['time'] += sa_comp_time
        results['SA']['A*'] = r_sa_astar

        r_sa_dijk = self.calculate_metrics(best_sol, 'dijkstra')
        r_sa_dijk['time'] += sa_comp_time
        results['SA']['Dijkstra'] = r_sa_dijk

        r_sa_bfs = self.calculate_metrics(best_sol, 'bfs')
        r_sa_bfs['time'] += sa_comp_time
        results['SA']['BFS'] = r_sa_bfs

        # --- 3. BRUTE FORCE (Düşünme Süresi) ---
        if len(self.items) <=11:
            t_bf_start = time.time()

            best_bf_order = None
            min_bf_dist = float('inf')
            # Permütasyon hesabı (Ağır işlem)
            for p in itertools.permutations(self.items):
                d = 0;
                c = self.start_pos
                for i in p: d += abs(i[0] - c[0]) + abs(i[1] - c[1]); c = i
                if d < min_bf_dist: min_bf_dist = d; best_bf_order = list(p)

            t_bf_end = time.time()
            bf_comp_time = (t_bf_end - t_bf_start) * 1000

            r_bf = self.calculate_metrics(best_bf_order, 'astar')
            r_bf['time'] += bf_comp_time
            results['BF'] = r_bf
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
        h1.addWidget(QLabel("Ürün Sayısı (Max 100):"))
        self.spin = QSpinBox();
        self.spin.setValue(6);
        self.spin.setRange(2, 100)
        h1.addWidget(self.spin)
        clayout.addLayout(h1)
        h2 = QHBoxLayout()
        b1 = QPushButton("Depo Üret");
        b1.clicked.connect(self.reset)
        b2 = QPushButton("Ürünleri Yerleştir");
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
        left_layout.addWidget(vis_group, stretch=3)

        right_layout = QVBoxLayout()
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)
        layout.addLayout(left_layout, stretch=5)
        layout.addLayout(right_layout, stretch=4)

    def reset(self):
        self.sim.generate_warehouse()
        self.figure.clear();
        self.canvas.draw()
        self.results_cache = None

    def order(self):
        self.sim.generate_order(self.spin.value())
        self.results_cache = None

    def run_calculations(self):
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

        ax1 = self.figure.add_subplot(221)  # Sol Üst
        ax3 = self.figure.add_subplot(222)  # Sağ Üst
        ax4 = self.figure.add_subplot(223)  # Sol Alt
        ax6 = self.figure.add_subplot(224)

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
        ax1.set_title('Mesafe')
        ax1.set_ylabel('Birim')
        for i, v in enumerate(dists): ax1.text(i, v, str(v), ha='center', va='bottom', fontsize=8, fontweight='bold')

        # 2. Algoritma Tutarlılığı
        labels = ['A*', 'Dij', 'BFS']
        x = np.arange(len(labels));
        width = 0.35

        # 3. Verimlilik
        knn_vis = [res['KNN']['A*']['vis'], res['KNN']['Dijkstra']['vis'], res['KNN']['BFS']['vis']]
        sa_vis = [res['SA']['A*']['vis'], res['SA']['Dijkstra']['vis'], res['SA']['BFS']['vis']]
        ax3.bar(x - width / 2, knn_vis, width, color='blue', alpha=0.6)
        ax3.bar(x + width / 2, sa_vis, width, color='green', alpha=0.6)
        ax3.set_title('Verimlilik (Gezilen Düğüm Sayısı)')
        ax3.set_xticks(x);
        ax3.set_xticklabels(labels)

        # -------------------------------------------------------------
        # 4. GRAFİK (DEĞİŞTİRİLEN KISIM): BRUTE FORCE SÜRESİ EKLENDİ
        # -------------------------------------------------------------
        # Verileri hazırla
        time_labels = ['K-A*', 'K-Dij', 'K-BFS', 'S-A*', 'S-Dij', 'S-BFS']
        time_values = [
            res['KNN']['A*']['time'], res['KNN']['Dijkstra']['time'], res['KNN']['BFS']['time'],
            res['SA']['A*']['time'], res['SA']['Dijkstra']['time'], res['SA']['BFS']['time']
        ]
        time_colors = ['blue', 'blue', 'blue', 'green', 'green', 'green']

        # Brute Force varsa listeye ekle
        if res['BF']:
            time_labels.append('BF')
            time_values.append(res['BF']['time'])
            time_colors.append('gold')

        # Çizim
        ax4.bar(time_labels, time_values, color=time_colors)
        ax4.set_title('Hesaplama Süresi (Logaritmik)', fontsize=10)
        ax4.set_ylabel('Milisaniye (ms)', fontsize=9)

        # LOGARİTMİK ÖLÇEK (Yoksa BF 2000 iken, diğerleri 0.1 görünmez)
        ax4.set_yscale('log')

        # X Eksen yazılarını küçült sığsın
        ax4.tick_params(axis='x', labelsize=8)

        # Değerleri yaz (Değer küçükse okunur formatta)
        for i, v in enumerate(time_values):
            txt = f"{v:.1f}" if v > 1 else f"{v:.2f}"
            ax4.text(i, v, txt, ha='center', va='bottom', fontsize=8, color='black')

        # -------------------------------------------------------------


        # 6. VERİMLİLİK PUANI
        comp_labels = []
        scores = []
        bar_colors = []

        # Formül: Sabit / (Mesafe * (Süre + 1)^2)
        def calculate_score(dist, time_ms):
            if dist == 0: return 0
            time_penalty = (time_ms + 1) ** 2
            return 10000000 / (dist * time_penalty)

        for sub in ['A*', 'Dijkstra', 'BFS']:
            comp_labels.append(f"KNN\n{sub}")
            s = calculate_score(res['KNN'][sub]['dist'], res['KNN'][sub]['time'])
            scores.append(s)
            bar_colors.append('lightblue')

        for sub in ['A*', 'Dijkstra', 'BFS']:
            comp_labels.append(f"SA\n{sub}")
            s = calculate_score(res['SA'][sub]['dist'], res['SA'][sub]['time'])
            scores.append(s)
            bar_colors.append('lightgreen')

        if res['BF']:
            comp_labels.append("BF")
            s = calculate_score(res['BF']['dist'], res['BF']['time'])
            scores.append(s)
            bar_colors.append('gold')

        score_bars = ax6.bar(comp_labels, scores, color=bar_colors, edgecolor='black', alpha=0.8)
        ax6.set_title('Verimlilik Puanı\n(Hesaplama Zamanı Dahil Edildi)', fontsize=9, fontweight='bold',
                      color='darkred')
        ax6.set_ylabel('Verimlilik Skoru', fontsize=9)
        ax6.tick_params(axis='x', labelsize=7, rotation=15)
        ax6.grid(axis='y', linestyle='--', alpha=0.3)

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