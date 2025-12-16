import sys
import random
import heapq
import math
import time
import numpy as np
from collections import deque
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QMessageBox,
                             QSpinBox, QGroupBox)
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# --- AYARLAR ---
GRID_SIZE = 30
OBSTACLE_DENSITY = 0.6


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
        return total_dist, total_visited, total_turns, full_path

    def run_full_analysis(self):
        if not self.items: return None
        results = {}

        random_state = random.getstate()
        random.seed(str(self.items))

        # A) K-NN
        t_start = time.time()
        unvisited = sorted(self.items.copy())
        curr = self.start_pos;
        knn_order = []
        while unvisited:
            nearest = min(unvisited, key=lambda x: abs(x[0] - curr[0]) + abs(x[1] - curr[1]))
            knn_order.append(nearest);
            unvisited.remove(nearest);
            curr = nearest
        dist, vis, turns, path = self.calculate_metrics(knn_order, algo_type='astar')
        results['KNN'] = {'dist': dist, 'time': (time.time() - t_start) * 1000, 'turns': turns, 'path': path,
                          'vis': vis}

        # B) Simulated Annealing
        t_start = time.time()
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

        dist, vis, turns, path = self.calculate_metrics(best_sol, algo_type='astar')
        results['SA'] = {'dist': dist, 'time': (time.time() - t_start) * 1000, 'turns': turns, 'path': path, 'vis': vis}

        # C) Rastgele
        t_start = time.time()
        rnd_items = sorted(self.items.copy());
        random.shuffle(rnd_items)
        dist, vis, turns, path = self.calculate_metrics(rnd_items, algo_type='astar')
        results['RND'] = {'dist': dist, 'time': (time.time() - t_start) * 1000, 'turns': turns, 'path': path,
                          'vis': vis}

        random.setstate(random_state)

        # D) Pathfinding Kıyaslama
        best_order = best_sol
        _, astar_vis, _, _ = self.calculate_metrics(best_order, algo_type='astar')
        _, dij_vis, _, _ = self.calculate_metrics(best_order, algo_type='dijkstra')
        _, bfs_vis, _, _ = self.calculate_metrics(best_order, algo_type='bfs')

        results['Pathfinding'] = {'A*': astar_vis, 'Dijkstra': dij_vis, 'BFS': bfs_vis}

        return results

    def clear_paths(self):
        self.active_paths = []
        self.update()

    def add_path(self, path, color, width):
        self.active_paths.append({'path': path, 'color': color, 'width': width})
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
                painter.drawRect(c * w, r * h, w, h)

        for path_data in self.active_paths:
            path = path_data['path']
            if not path: continue
            pen = QPen(path_data['color'], path_data['width'])
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
        self.setWindowTitle("Depo Analiz: Optimizasyon Başarısı (ROI)")
        self.setGeometry(50, 50, 1400, 850)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        left_layout = QVBoxLayout()
        self.sim = WarehouseSimulation()
        left_layout.addWidget(self.sim, stretch=6)

        ctrl = QGroupBox("Kontroller")
        clayout = QVBoxLayout()
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Sipariş Sayısı:"));
        self.spin = QSpinBox();
        self.spin.setValue(8);
        h1.addWidget(self.spin)
        clayout.addLayout(h1)
        h2 = QHBoxLayout()
        b1 = QPushButton("Depo Üret");
        b1.clicked.connect(self.reset)
        b2 = QPushButton("Sipariş Ver");
        b2.clicked.connect(self.order)
        b3 = QPushButton("HESAPLA");
        b3.clicked.connect(self.run)
        b3.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        h2.addWidget(b1);
        h2.addWidget(b2);
        h2.addWidget(b3)
        clayout.addLayout(h2)
        lbl_info = QLabel("Mavi: K-NN | Yeşil: SA")
        lbl_info.setStyleSheet("font-weight: bold; color: #333;")
        clayout.addWidget(lbl_info)
        ctrl.setLayout(clayout)
        left_layout.addWidget(ctrl, stretch=1)

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

    def order(self):
        self.sim.generate_order(self.spin.value())
        QMessageBox.information(self, "Bilgi", "Ürünler yerleştirildi.")

    def run(self):
        res = self.sim.run_full_analysis()
        if not res: return

        self.sim.clear_paths()
        self.sim.add_path(res['KNN']['path'], QColor(0, 0, 255, 100), 6)
        self.sim.add_path(res['SA']['path'], QColor(0, 255, 0, 255), 2)

        self.figure.clear()

        ax1 = self.figure.add_subplot(231)
        ax2 = self.figure.add_subplot(232)
        ax3 = self.figure.add_subplot(233)
        ax4 = self.figure.add_subplot(234)
        ax5 = self.figure.add_subplot(235)

        algos = ['RND', 'K-NN', 'SA']
        colors = ['gray', 'blue', 'green']

        # 1. Mesafe
        ax1.bar(algos, [res['RND']['dist'], res['KNN']['dist'], res['SA']['dist']], color=colors)
        ax1.set_title('Toplam Mesafe')

        # 2. Verimlilik (A* vs Dijkstra vs BFS)
        pf_names = ['A*', 'Dijkstra', 'BFS']
        pf_vals = [res['Pathfinding']['A*'], res['Pathfinding']['Dijkstra'], res['Pathfinding']['BFS']]
        pf_colors = ['green', 'red', 'orange']
        ax2.bar(pf_names, pf_vals, color=pf_colors)
        ax2.set_title('İncelenen Düğüm (Efficiency)')
        ax2.set_ylabel('Kare')
        for i, v in enumerate(pf_vals):
            ax2.text(i, v, str(v), ha='center', va='bottom', fontsize=8, fontweight='bold')

        # 3. Süre
        ax3.plot(algos, [res['RND']['time'], res['KNN']['time'], res['SA']['time']], marker='o', color='purple')
        ax3.set_title('Süre (ms)')
        ax3.grid(True)

        # 4. Dönüş
        ax4.bar(algos, [res['RND']['turns'], res['KNN']['turns'], res['SA']['turns']], color=['gray', 'cyan', 'orange'])
        ax4.set_title('Dönüş Sayısı')

        # 5. YENİ GRAFİK: OPTİMİZASYON KAZANCI (%)
        # Rastgele (Random) rotaya göre ne kadar iyileştirme yapıldı?
        rnd_dist = res['RND']['dist']
        if rnd_dist > 0:
            knn_gain = (rnd_dist - res['KNN']['dist']) / rnd_dist * 100
            sa_gain = (rnd_dist - res['SA']['dist']) / rnd_dist * 100
        else:
            knn_gain = sa_gain = 0

        gains = [knn_gain, sa_gain]
        gain_labels = ['K-NN', 'SA']
        bars = ax5.bar(gain_labels, gains, color=['blue', 'green'])
        ax5.set_title('Optimizasyon Kazancı (%)')
        ax5.set_ylabel('% Tasarruf')
        ax5.set_ylim(0, 100)  # 0 ile 100 arası sabitle

        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width() / 2., height,
                     f'%{height:.1f}', ha='center', va='bottom', fontweight='bold')

        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())