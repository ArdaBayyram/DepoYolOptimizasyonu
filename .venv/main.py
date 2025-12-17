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
                             QSpinBox, QGroupBox, QCheckBox, QGridLayout, QRadioButton, QButtonGroup)
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush
from PyQt5.QtCore import Qt, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class YolBulucu:
    def __init__(self, grid):
        self.grid = grid
        self.update_dims()

    def update_dims(self):
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])

    def heuristic(self, a, b, use_heuristic):
        return (abs(a[0] - b[0]) + abs(a[1] - b[1])) if use_heuristic else 0

    def komsu_bul(self, node):
        komsu = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = node[0] + dx, node[1] + dy
            if 0 <= nx < self.rows and 0 <= ny < self.cols:
                if self.grid[nx][ny] == 0:
                    komsu.append((nx, ny))
        return komsu

    def agirlikliyolbul(self, start, goal, use_heuristic=True):
        open_set = []
        heapq.heappush(open_set, (0, 0, start))
        geldigi = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal, use_heuristic)}
        gezilen_dugum = set()
        sayac = 1

        while open_set:
            current = heapq.heappop(open_set)[2]
            gezilen_dugum.add(current)

            if current == goal:
                path = []
                while current in geldigi:
                    path.append(current)
                    current = geldigi[current]
                path.append(start)
                return path[::-1], len(gezilen_dugum)

            for neighbor in self.komsu_bul(current):
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    geldigi[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    h = self.heuristic(neighbor, goal, use_heuristic)
                    f_score[neighbor] = tentative_g_score + h
                    heapq.heappush(open_set, (f_score[neighbor], sayac, neighbor))
                    sayac += 1
        return [], len(gezilen_dugum)

    def bfsyolbul(self, start, goal):
        queue = deque([start])
        geldigiDugum = {start: None}
        gezilenDugum = set([start])

        while queue:
            current = queue.popleft()
            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = geldigiDugum[current]
                return path[::-1], len(gezilenDugum)

            for neighbor in self.komsu_bul(current):
                if neighbor not in gezilenDugum:
                    gezilenDugum.add(neighbor)
                    geldigiDugum[neighbor] = current
                    queue.append(neighbor)
        return [], len(gezilenDugum)


class DepoSim(QWidget):
    itemCountChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = 55
        self.cols = 55
        self.grid = np.zeros((self.rows, self.cols))
        self.items = []
        self.start_pos = (0, 0)
        self.active_paths = []
        self.pathfinder = None
        self.edit_mode = 0
        self.DepoOlustur(self.rows, self.cols)

    def DepoOlustur(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols))
        self.items = []

        c = 1
        while c < cols - 1:
            if abs(c - cols // 2) < 2:
                c += 1
                continue
            if random.random() < 0.7:
                shelf_width = 1
                if c < cols - 2 and random.random() < 0.3: shelf_width = 2
                r = 1
                while r < rows - 1:
                    block_len = random.randint(3, 8)
                    if r + block_len >= rows - 1: block_len = rows - 1 - r
                    for k in range(block_len):
                        self.grid[r + k][c] = 1
                        if shelf_width == 2: self.grid[r + k][c + 1] = 1
                    r += block_len
                    gap_len = random.randint(1, 3)
                    r += gap_len
                c += shelf_width + 1
            else:
                c += 1

        mid_col = cols // 2
        safe_r = 0
        while safe_r < rows and self.grid[safe_r][mid_col] == 1:
            safe_r += 1
        if safe_r >= rows: safe_r = 0
        self.start_pos = (safe_r, mid_col)

        self.pathfinder = YolBulucu(self.grid)
        self.active_paths = []
        self.itemCountChanged.emit(0)
        self.update()

    def mousePressEvent(self, event):
        w = self.width() / self.cols
        h = self.height() / self.rows
        c = int(event.x() / w)
        r = int(event.y() / h)

        if 0 <= r < self.rows and 0 <= c < self.cols:
            if self.edit_mode == 1:
                if (r, c) != self.start_pos and (r, c) not in self.items:
                    self.grid[r][c] = 1 - self.grid[r][c]
                    self.pathfinder.grid = self.grid
                    self.pathfinder.update_dims()
            elif self.edit_mode == 2:
                if self.grid[r][c] == 0 and (r, c) != self.start_pos:
                    if (r, c) in self.items:
                        self.items.remove((r, c))
                    else:
                        self.items.append((r, c))
                    self.itemCountChanged.emit(len(self.items))
            self.update()

    def olusturmaduzeni(self, count):
        self.items = []
        attempts = 0
        while len(self.items) < count and attempts < 5000:
            rx, ry = random.randint(0, self.rows - 1), random.randint(0, self.cols - 1)
            if self.grid[rx][ry] == 0 and (rx, ry) != self.start_pos:
                if (rx, ry) not in self.items:
                    self.items.append((rx, ry))
            attempts += 1
        self.active_paths = []
        self.itemCountChanged.emit(len(self.items))
        self.update()

    def hesaplama(self, order, algo_type='astar'):
        toplamMesafe = 0;
        toplamGidilen = 0;
        toplamDonus = 0
        ButunMesafe = []
        curr = self.start_pos
        last_dx, last_dy = 0, 0

        t_start = time.time()

        for item in order:
            if algo_type == 'astar':
                yol, gidilen = self.pathfinder.agirlikliyolbul(curr, item, use_heuristic=True)
            elif algo_type == 'dijkstra':
                yol, gidilen = self.pathfinder.agirlikliyolbul(curr, item, use_heuristic=False)
            elif algo_type == 'bfs':
                yol, gidilen = self.pathfinder.bfsyolbul(curr, item)
            else:
                yol, gidilen = [], 0

            toplamGidilen += gidilen
            if len(yol) > 1:
                toplamMesafe += len(yol) - 1
                for i in range(len(yol) - 1):
                    p1, p2 = yol[i], yol[i + 1]
                    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                    if (last_dx, last_dy) != (0, 0) and (dx, dy) != (last_dx, last_dy):
                        toplamDonus += 1
                    last_dx, last_dy = dx, dy

                if not ButunMesafe:
                    ButunMesafe.extend(yol)
                else:
                    ButunMesafe.extend(yol[1:])
            curr = item

        t_end = time.time()

        return {
            'dist': toplamMesafe,
            'vis': toplamGidilen,
            'turns': toplamDonus,
            'time': (t_end - t_start) * 1000,
            'path': ButunMesafe
        }

    def analiz(self, n_tests=1):
        if not self.items: return None

        agg_results = {'KNN': {'A*': [], 'Dijkstra': [], 'BFS': []},
                       'SA': {'A*': [], 'Dijkstra': [], 'BFS': []},
                       'RND': {'A*': []}, 'BF': []}

        final_best_paths = {'KNN': {}, 'SA': {}, 'RND': {}, 'BF': None}

        random_state_init = random.getstate()

        for _ in range(n_tests):

            t_knn_start = time.time()
            gidilmeyen = sorted(self.items.copy())
            curr = self.start_pos
            knn_order = []
            while gidilmeyen:
                nearest = min(gidilmeyen, key=lambda x: abs(x[0] - curr[0]) + abs(x[1] - curr[1]))
                knn_order.append(nearest)
                gidilmeyen.remove(nearest)
                curr = nearest
            t_knn_end = time.time()
            knn_comp_time = (t_knn_end - t_knn_start) * 1000

            for sub in ['astar', 'dijkstra', 'bfs']:
                key_map = {'astar': 'A*', 'dijkstra': 'Dijkstra', 'bfs': 'BFS'}
                r = self.hesaplama(knn_order, sub)
                r['time'] += knn_comp_time
                agg_results['KNN'][key_map[sub]].append(r)
                if not final_best_paths['KNN'].get(key_map[sub]): final_best_paths['KNN'][key_map[sub]] = r

            t_sa_start = time.time()
            curr_sol = sorted(self.items.copy())
            random.shuffle(curr_sol)

            def maliyet(sol):
                d = 0;
                c = self.start_pos
                for i in sol: d += abs(i[0] - c[0]) + abs(i[1] - c[1]); c = i
                return d

            best_sol = list(curr_sol)
            best_cost = maliyet(curr_sol)
            temp = 1000.0
            for _ in range(1000):
                new_sol = list(curr_sol)
                i1, i2 = random.sample(range(len(new_sol)), 2)
                new_sol[i1], new_sol[i2] = new_sol[i2], new_sol[i1]
                current_c = maliyet(curr_sol)
                new_c = maliyet(new_sol)
                if new_c < current_c or random.random() < math.exp((current_c - new_c) / temp):
                    curr_sol = new_sol
                    if maliyet(curr_sol) < best_cost: best_sol = list(curr_sol); best_cost = maliyet(curr_sol)
                temp *= 0.99
            t_sa_end = time.time()
            sa_comp_time = (t_sa_end - t_sa_start) * 1000

            for sub in ['astar', 'dijkstra', 'bfs']:
                key_map = {'astar': 'A*', 'dijkstra': 'Dijkstra', 'bfs': 'BFS'}
                r = self.hesaplama(best_sol, sub)
                r['time'] += sa_comp_time
                agg_results['SA'][key_map[sub]].append(r)
                if not final_best_paths['SA'].get(key_map[sub]) or r['dist'] < final_best_paths['SA'][key_map[sub]][
                    'dist']:
                    final_best_paths['SA'][key_map[sub]] = r

            if len(self.items) <= 11:
                t_bf_start = time.time()
                bestBForder = None
                minBForder = float('inf')
                for p in itertools.permutations(self.items):
                    d = 0;
                    c = self.start_pos
                    for i in p: d += abs(i[0] - c[0]) + abs(i[1] - c[1]); c = i
                    if d < minBForder: minBForder = d; bestBForder = list(p)
                t_bf_end = time.time()
                bf_comp_time = (t_bf_end - t_bf_start) * 1000
                r_bf = self.hesaplama(bestBForder, 'astar')
                r_bf['time'] += bf_comp_time
                agg_results['BF'].append(r_bf)
                final_best_paths['BF'] = r_bf
            else:
                pass

        random.setstate(random_state_init)

        avg_results = {'KNN': {}, 'SA': {}, 'RND': {}, 'BF': None}

        def avg_dict(list_of_dicts):
            if not list_of_dicts: return None
            base = list_of_dicts[0].copy()
            for k in ['dist', 'vis', 'turns', 'time']:
                base[k] = sum(d[k] for d in list_of_dicts) / len(list_of_dicts)
            return base

        for cat in ['KNN', 'SA']:
            for sub in ['A*', 'Dijkstra', 'BFS']:
                res = avg_dict(agg_results[cat][sub])
                res['path'] = final_best_paths[cat][sub]['path']
                res['best_dist'] = final_best_paths[cat][sub]['dist']
                avg_results[cat][sub] = res

        if agg_results['BF']:
            res = avg_dict(agg_results['BF'])
            res['path'] = final_best_paths['BF']['path']
            res['best_dist'] = final_best_paths['BF']['dist']
            avg_results['BF'] = res

        return avg_results

    def rotaTemizle(self):
        self.active_paths = []
        self.update()

    def rotaEkle(self, path, color, width, style=Qt.SolidLine):
        self.active_paths.append({'path': path, 'color': color, 'width': width, 'style': style})
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        w = self.width() / self.cols
        h = self.height() / self.rows

        for r in range(self.rows):
            for c in range(self.cols):
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Depo Ürün Toplama Optimizasyonu")
        self.setGeometry(50, 50, 1500, 900)
        self.results_cache = None

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        left_layout = QVBoxLayout()
        self.sim = DepoSim()
        left_layout.addWidget(self.sim, stretch=6)

        ctrl = QGroupBox("Kontroller")
        clayout = QVBoxLayout()

        dim_layout = QHBoxLayout()
        dim_layout.addWidget(QLabel("Genişlik:"))
        self.spin_w = QSpinBox()
        self.spin_w.setRange(10, 200)
        self.spin_w.setValue(60)
        dim_layout.addWidget(self.spin_w)
        dim_layout.addWidget(QLabel("Yükseklik:"))
        self.spin_h = QSpinBox()
        self.spin_h.setRange(10, 200)
        self.spin_h.setValue(60)
        dim_layout.addWidget(self.spin_h)
        clayout.addLayout(dim_layout)

        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Ürün:"))
        self.spin = QSpinBox();
        self.spin.setValue(5);
        self.spin.setRange(0, 100)
        h1.addWidget(self.spin)
        h1.addWidget(QLabel("Test N:"))
        self.spin_n = QSpinBox()
        self.spin_n.setValue(1)
        self.spin_n.setRange(1, 100)
        h1.addWidget(self.spin_n)
        clayout.addLayout(h1)

        mode_layout = QHBoxLayout()
        self.rb_none = QRadioButton("Gezin")
        self.rb_obs = QRadioButton("Engel")
        self.rb_item = QRadioButton("Eşya")
        self.rb_none.setChecked(True)
        self.bg = QButtonGroup()
        self.bg.addButton(self.rb_none, 0)
        self.bg.addButton(self.rb_obs, 1)
        self.bg.addButton(self.rb_item, 2)
        self.bg.buttonClicked[int].connect(self.change_mode)
        mode_layout.addWidget(self.rb_none)
        mode_layout.addWidget(self.rb_obs)
        mode_layout.addWidget(self.rb_item)
        clayout.addLayout(mode_layout)

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

        left_layout.addWidget(ctrl, stretch=2)

        vis_group = QGroupBox("Görselleştirme")
        vis_layout = QGridLayout()
        self.check_knn_astar = QCheckBox("K-NN + A* (Mavi)");
        self.check_knn_astar.setChecked(True)
        self.check_knn_dijk = QCheckBox("K-NN + Dij (Turkuaz)")
        self.check_knn_bfs = QCheckBox("K-NN + BFS (Mor)")
        self.check_sa_astar = QCheckBox("SA + A* (Yeşil)");
        self.check_sa_astar.setChecked(True)
        self.check_sa_dijk = QCheckBox("SA + Dij (Gri)")
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

        left_layout.addWidget(vis_group, stretch=2)

        right_layout = QVBoxLayout()
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)
        layout.addLayout(left_layout, stretch=5)
        layout.addLayout(right_layout, stretch=4)

        self.sim.itemCountChanged.connect(self.spin.setValue)

    def change_mode(self, id):
        self.sim.edit_mode = id

    def reset(self):
        self.sim.DepoOlustur(self.spin_h.value(), self.spin_w.value())
        self.figure.clear();
        self.canvas.draw()
        self.results_cache = None

    def order(self):
        self.sim.olusturmaduzeni(self.spin.value())
        self.results_cache = None

    def run_calculations(self):
        if len(self.sim.items) > 11:
            if self.check_bf.isChecked():
                QMessageBox.warning(self, "Sınır Uyarısı",
                                    "Ürün sayısı 11'den fazla!\nBrute Force işlemi çok uzun süreceği için iptal edilecek.")
                self.check_bf.setChecked(False)

        self.results_cache = self.sim.analiz(self.spin_n.value())
        if not self.results_cache: return
        self.update_visualization()
        self.draw_graphs()

    def update_visualization(self):
        if not self.results_cache: return
        self.sim.rotaTemizle()
        res = self.results_cache

        if self.check_knn_astar.isChecked(): self.sim.rotaEkle(res['KNN']['A*']['path'], QColor(0, 0, 255, 120), 6)
        if self.check_knn_dijk.isChecked(): self.sim.rotaEkle(res['KNN']['Dijkstra']['path'], QColor(0, 90, 255, 120),
                                                              4, Qt.DashLine)
        if self.check_knn_bfs.isChecked(): self.sim.rotaEkle(res['KNN']['BFS']['path'], QColor(128, 0, 128, 120), 2,
                                                             Qt.DotLine)

        if self.check_sa_astar.isChecked(): self.sim.rotaEkle(res['SA']['A*']['path'], QColor(0, 255, 0, 200), 3)
        if self.check_sa_dijk.isChecked(): self.sim.rotaEkle(res['SA']['Dijkstra']['path'], QColor(35, 70, 40, 200),
                                                             2, Qt.DashLine)
        if self.check_sa_bfs.isChecked(): self.sim.rotaEkle(res['SA']['BFS']['path'], QColor(255, 215, 0, 200), 2,
                                                            Qt.DotLine)

        if self.check_bf.isChecked() and res['BF']:
            self.sim.rotaEkle(res['BF']['path'], QColor(255, 215, 0, 255), 4)

    def draw_graphs(self):
        self.figure.clear()
        res = self.results_cache

        ax1 = self.figure.add_subplot(221)
        ax3 = self.figure.add_subplot(222)
        ax4 = self.figure.add_subplot(223)
        ax6 = self.figure.add_subplot(224)

        algos = ['K-NN', 'SA']
        dists = [res['KNN']['A*']['best_dist'], res['SA']['A*']['best_dist']]
        colors = ['blue', 'green']

        if res['BF']:
            algos.append('BF (Opt)')
            dists.append(res['BF']['best_dist'])
            colors.append('gold')

        ax1.bar(algos, dists, color=colors)
        ax1.set_title(f'En İyi Mesafe (N={self.spin_n.value()})')
        ax1.set_ylabel('Birim')
        for i, v in enumerate(dists): ax1.text(i, v, f"{v:.1f}", ha='center', va='bottom', fontsize=8,
                                               fontweight='bold')

        labels = ['A*', 'Dij', 'BFS']
        x = np.arange(len(labels));
        width = 0.35

        knn_vis = [res['KNN']['A*']['vis'], res['KNN']['Dijkstra']['vis'], res['KNN']['BFS']['vis']]
        sa_vis = [res['SA']['A*']['vis'], res['SA']['Dijkstra']['vis'], res['SA']['BFS']['vis']]
        ax3.bar(x - width / 2, knn_vis, width, color='blue', alpha=0.6)
        ax3.bar(x + width / 2, sa_vis, width, color='green', alpha=0.6)
        ax3.set_title('Ortalama Verimlilik')
        ax3.set_xticks(x);
        ax3.set_xticklabels(labels)

        time_labels = ['K-A*', 'K-Dij', 'K-BFS', 'S-A*', 'S-Dij', 'S-BFS']
        time_values = [
            res['KNN']['A*']['time'], res['KNN']['Dijkstra']['time'], res['KNN']['BFS']['time'],
            res['SA']['A*']['time'], res['SA']['Dijkstra']['time'], res['SA']['BFS']['time']
        ]
        time_colors = ['blue', 'blue', 'blue', 'green', 'green', 'green']

        if res['BF']:
            time_labels.append('BF')
            time_values.append(res['BF']['time'])
            time_colors.append('gold')

        ax4.bar(time_labels, time_values, color=time_colors)
        ax4.set_title('Ortalama Süre (ms)', fontsize=10)
        ax4.set_ylabel('ms', fontsize=9)

        ax4.set_yscale('log')

        ax4.tick_params(axis='x', labelsize=8)

        for i, v in enumerate(time_values):
            txt = f"{v:.1f}" if v > 1 else f"{v:.2f}"
            ax4.text(i, v, txt, ha='center', va='bottom', fontsize=8, color='black')

        comp_labels = []
        scores = []
        bar_colors = []

        def calculate_score(dist, time_ms):
            if dist == 0: return 0
            time_penalty = (time_ms + 1)
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
        ax6.set_title('Puan', fontsize=9, fontweight='bold',
                      color='darkred')
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