import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd

def open_scoring_import_hub(self):
    """Fenêtre pour importer un scoring depuis Excel/CSV ou EDF Hypnogram."""
    if not self.raw:
        messagebox.showwarning("Attention", "Veuillez d'abord charger un fichier EDF")
        return
    hub = tk.Toplevel(self.root)
    hub.title("Importer Scoring (Excel/EDF)")
    hub.geometry("420x200")
    hub.transient(self.root)
    hub.grab_set()
    frame = ttk.Frame(hub, padding=12)
    frame.pack(fill=tk.BOTH, expand=True)
    ttk.Label(frame, text="Choisissez la source de scoring à importer:", font=('Helvetica', 10, 'bold')).pack(anchor='w')
    ttk.Button(frame, text="Importer Excel/CSV", command=lambda: (hub.destroy(), self._import_manual_scoring_excel())).pack(fill=tk.X, pady=(12,6))
    ttk.Button(frame, text="Charger Hypnogram EDF (Sleep-EDFx)", command=lambda: (hub.destroy(), self._load_hypnogram_edfplus())).pack(fill=tk.X)

def open_manual_scoring_editor(self):
    """Éditeur simple pour scorer manuellement les époques visibles ou par saisie."""
    if not self.raw:
        messagebox.showwarning("Attention", "Veuillez d'abord charger un fichier EDF")
        return
    win = tk.Toplevel(self.root)
    win.title("Éditeur de Scoring Manuel")
    win.geometry("1100x900")
    win.transient(self.root)
    win.grab_set()
    cols = ("time", "stage")
    tree = ttk.Treeview(win, columns=cols, show="headings")
    for c in cols:
        tree.heading(c, text=c)
        tree.column(c, width=120, anchor=tk.CENTER)
    tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
    df_init = None
    if getattr(self, 'manual_scoring_data', None) and len(self.manual_scoring_data) > 0:
        df_init = self.manual_scoring_data
    elif getattr(self, 'sleep_scoring_data', None) and len(self.sleep_scoring_data) > 0:
        df_init = self.sleep_scoring_data
    if df_init is not None:
        try:
            for _, row in df_init.iterrows():
                tree.insert('', tk.END, values=(float(row['time']), str(row['stage'])))
        except Exception:
            pass

    form = ttk.Frame(win)
    form.pack(fill=tk.X, padx=8, pady=(0,8))
    ttk.Label(form, text="time (s)").grid(row=0, column=0, sticky='w')
    time_var = tk.DoubleVar(value=float(getattr(self, 'current_time', 0.0)))
    ttk.Entry(form, textvariable=time_var, width=10).grid(row=0, column=1, sticky='w', padx=(4,10))
    ttk.Label(form, text="stage").grid(row=0, column=2, sticky='w')
    stage_var = tk.StringVar(value='W')
    ttk.Combobox(form, textvariable=stage_var, values=['W','N1','N2','N3','R','U'], width=6).grid(row=0, column=3, sticky='w', padx=(4,10))
    ttk.Button(form, text="Ajouter/Mettre à jour", command=lambda: add_or_update(tree, time_var, stage_var)).grid(row=0, column=4, padx=(6,0))
    ttk.Button(form, text="Supprimer sélection", command=lambda: delete_selected(tree)).grid(row=0, column=5, padx=(6,0))

def add_or_update(tree, time_var, stage_var):
    # Implémenter la logique d’ajout ou modification dans le Treeview
    pass

def delete_selected(tree):
    # Implémenter la suppression de l’élément sélectionné dans le Treeview
    pass

def save_active_scoring(self):
    """Sauvegarde le scoring actif (manuel prioritaire) en CSV."""
    df = get_active_scoring_df(self)
    if df is None or df.empty:
        messagebox.showwarning("Scoring", "Aucun scoring à sauvegarder")
        return
    file_path = filedialog.asksaveasfilename(title="Sauvegarder scoring (CSV)", defaultextension=".csv",
                                             filetypes=[("CSV", "*.csv")])
    if not file_path:
        return
    try:
        df[['time', 'stage']].to_csv(file_path, index=False, encoding='utf-8')
        self.scoring_dirty = False
        messagebox.showinfo("Scoring", f"Scoring sauvegardé: {file_path}")
    except Exception as e:
        messagebox.showerror("Erreur", f"Echec sauvegarde scoring: {e}")

def get_active_scoring_df(self):
    """Retourne le DataFrame du scoring actif (manuel prioritaire)."""
    if hasattr(self, 'manual_scoring_data') and self.manual_scoring_data is not None and len(self.manual_scoring_data):
        return self.manual_scoring_data
    elif hasattr(self, 'sleep_scoring_data') and self.sleep_scoring_data is not None and len(self.sleep_scoring_data):
        return self.sleep_scoring_data
    return None

def export_periods_and_metrics_csv(self, data, filename):
    """
    Exporter périodes et métriques dans un fichier CSV.
    Args:
        data (pd.DataFrame): données à exporter
        filename (str): chemin du fichier
    """
    try:
        data.to_csv(filename, index=False, encoding='utf-8')
        messagebox.showinfo("Export", f"Export réussi vers {filename}")
    except Exception as e:
        messagebox.showerror("Erreur Export", f"Erreur lors de l'export: {e}")

def analyze_sleep_periods(self):
    """Analyse des périodes de sommeil (type SleepEEGpy)."""
    # Utiliser le scoring manuel s'il est non vide, sinon auto
    df = self._get_active_scoring_df()
    if df is None or len(df) == 0:
        messagebox.showwarning("Avertissement", "Aucun scoring de sommeil chargé.")
        return

    try:
        epoch_len = float(getattr(self, 'scoring_epoch_duration', 30.0))
        stages = df['stage'].astype(str).str.upper().values
        times = df['time'].values

        # Basic metrics (SleepEEGpy-like)
        t_start = float(times.min())
        t_end = float(times.max() + epoch_len)
        tib = (t_end - t_start) / 60.0  # Time in bed (min)

        # Sleep onset latency (first epoch in sleep N1/N2/N3/R)
        sleep_mask = np.isin(stages, ['N1', 'N2', 'N3', 'R'])
        if sleep_mask.any():
            first_sleep_idx = int(np.where(sleep_mask)[0][0])
            sol = (float(times[first_sleep_idx]) - t_start) / 60.0
        else:
            sol = float('nan')

        # Time asleep and WASO
        tst_sec = int(np.sum(np.isin(stages, ['N1','N2','N3','R'])) * epoch_len)
        waso_sec = int(np.sum(stages == 'W') * epoch_len)
        se = (tst_sec / (tib * 60.0)) * 100.0 if tib > 0 else float('nan')

        # REM latency from sleep onset
        if sleep_mask.any() and np.any(stages == 'R'):
            rem_idxs = np.where(stages == 'R')[0]
            rem_lat = (float(times[rem_idxs[0]]) - float(times[first_sleep_idx])) / 60.0
        else:
            rem_lat = float('nan')

        # Stage durations (min)
        stage_durations_min = {
            s: (int(np.sum(stages == s)) * epoch_len) / 60.0 for s in ['W', 'N1', 'N2', 'N3', 'R']
        }

        # Awakenings: count transitions into W with at least 1 epoch
        awakenings = 0
        for i in range(1, len(stages)):
            if stages[i] == 'W' and stages[i-1] != 'W':
                awakenings += 1

        # Build contiguous periods (start-end in minutes, label)
        periods = []
        if len(stages) > 0:
            start_idx = 0
            for i in range(1, len(stages)):
                if stages[i] != stages[i-1]:
                    periods.append((float(times[start_idx]) / 60.0, float(times[i]) / 60.0, stages[i-1]))
                    start_idx = i
            # last
            periods.append((float(times[start_idx]) / 60.0, float((times[-1] + epoch_len)) / 60.0, stages[-1]))

        # UI: window with summary + table of periods
        top = tk.Toplevel(self.root)
        top.title("Analyse des périodes de sommeil (type SleepEEGpy)")
        top.geometry("1100x900")
        top.transient(self.root)
        top.grab_set()

        container = ttk.Frame(top, padding=10)
        container.pack(fill=tk.BOTH, expand=True)

        summary = (
            f"TIB: {tib:.1f} min | SOL: {sol:.1f} min | TST: {tst_sec/60.0:.1f} min\n"
            f"WASO: {waso_sec/60.0:.1f} min | SE: {se:.1f}% | REM lat.: {rem_lat:.1f} min\n"
            f"W: {stage_durations_min['W']:.1f} | N1: {stage_durations_min['N1']:.1f} | N2: {stage_durations_min['N2']:.1f} | "
            f"N3: {stage_durations_min['N3']:.1f} | R: {stage_durations_min['R']:.1f} | Réveils: {awakenings}"
        )
        ttk.Label(container, text=summary, font=('Helvetica', 10)).pack(anchor='w', pady=(0,10))

        # Boutons d'export
        btns = ttk.Frame(container)
        btns.pack(fill=tk.X, pady=(0,10))
        ttk.Button(btns, text="Exporter CSV", command=lambda: _export_periods_and_metrics_csv(periods, tib, sol, tst_sec, waso_sec, se, rem_lat, stage_durations_min, awakenings)).pack(side=tk.RIGHT)
        ttk.Button(btns, text="Enregistrer Figure", command=lambda: _save_periods_figure(tree)).pack(side=tk.RIGHT, padx=(10,0))

        cols = ("Début (min)", "Fin (min)", "Stade")
        tree = ttk.Treeview(container, columns=cols, show="headings")
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=120, anchor=tk.CENTER)
        tree.pack(fill=tk.BOTH, expand=True)

        for (a, b, s) in periods:
            tree.insert("", tk.END, values=(f"{a:.1f}", f"{b:.1f}", s))

        def _export_periods_and_metrics_csv(periods_list, tib_min, sol_min, tst_seconds, waso_seconds, se_pct, rem_latency_min, stage_dur_min_dict, n_awaken):
            try:
                file_path = filedialog.asksaveasfilename(title="Exporter CSV", defaultextension=".csv",
                                                            filetypes=[("CSV", "*.csv")])
                if not file_path:
                    return
                import csv
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # Metrics
                    writer.writerow(["metric", "value"]) 
                    writer.writerow(["TIB_min", f"{tib_min:.2f}"])
                    writer.writerow(["SOL_min", f"{sol_min:.2f}"])
                    writer.writerow(["TST_min", f"{tst_seconds/60.0:.2f}"])
                    writer.writerow(["WASO_min", f"{waso_seconds/60.0:.2f}"])
                    writer.writerow(["SE_percent", f"{se_pct:.2f}"])
                    writer.writerow(["REM_latency_min", f"{rem_latency_min:.2f}"])
                    for sname, minutes in stage_dur_min_dict.items():
                        writer.writerow([f"Duration_{sname}_min", f"{minutes:.2f}"])
                    writer.writerow(["Awakenings", n_awaken])
                    writer.writerow([])
                    # Periods table
                    writer.writerow(["start_min", "end_min", "stage"]) 
                    for (a, b, s) in periods_list:
                        writer.writerow([f"{a:.2f}", f"{b:.2f}", s])
            except Exception as e:
                messagebox.showerror("Erreur", f"Echec de l'export CSV: {e}")

        def _save_periods_figure(treeview):
            try:
                # Simple export: capture de la fenêtre via matplotlib n'est pas direct; on exporte la liste en figure.
                import matplotlib.pyplot as _plt
                _fig, _ax = _plt.subplots(figsize=(6, len(periods)/6 + 1))
                y = 0
                for (a, b, s) in periods:
                    _ax.plot([a, b], [y, y], lw=6)
                    _ax.text(b + 0.2, y, s)
                    y += 1
                _ax.set_xlabel("Temps (min)")
                _ax.set_title("Périodes de sommeil")
                _ax.set_yticks([])
                _plt.tight_layout()
                file_path = filedialog.asksaveasfilename(title="Enregistrer la figure", defaultextension=".png",
                                                            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
                if file_path:
                    _fig.savefig(file_path, dpi=200, bbox_inches='tight')
                _plt.close(_fig)
            except Exception as e:
                messagebox.showerror("Erreur", f"Echec de l'enregistrement de la figure: {e}")
    except Exception as e:
        messagebox.showerror("Erreur", f"Echec analyse périodes: {e}")


def _analyze_sleep_periods(self):
    """Analyse des périodes de sommeil (type SleepEEGpy)."""
    # Utiliser le scoring manuel s'il est non vide, sinon auto
    df = self._get_active_scoring_df()
    if df is None or len(df) == 0:
        messagebox.showwarning("Avertissement", "Aucun scoring de sommeil chargé.")
        return

    try:
        epoch_len = float(getattr(self, 'scoring_epoch_duration', 30.0))
        stages = df['stage'].astype(str).str.upper().values
        times = df['time'].values

        # Basic metrics (SleepEEGpy-like)
        t_start = float(times.min())
        t_end = float(times.max() + epoch_len)
        tib = (t_end - t_start) / 60.0  # Time in bed (min)

        # Sleep onset latency (first epoch in sleep N1/N2/N3/R)
        sleep_mask = np.isin(stages, ['N1', 'N2', 'N3', 'R'])
        if sleep_mask.any():
            first_sleep_idx = int(np.where(sleep_mask)[0][0])
            sol = (float(times[first_sleep_idx]) - t_start) / 60.0
        else:
            sol = float('nan')

        # Time asleep and WASO
        tst_sec = int(np.sum(np.isin(stages, ['N1','N2','N3','R'])) * epoch_len)
        waso_sec = int(np.sum(stages == 'W') * epoch_len)
        se = (tst_sec / (tib * 60.0)) * 100.0 if tib > 0 else float('nan')

        # REM latency from sleep onset
        if sleep_mask.any() and np.any(stages == 'R'):
            rem_idxs = np.where(stages == 'R')[0]
            rem_lat = (float(times[rem_idxs[0]]) - float(times[first_sleep_idx])) / 60.0
        else:
            rem_lat = float('nan')

        # Stage durations (min)
        stage_durations_min = {
            s: (int(np.sum(stages == s)) * epoch_len) / 60.0 for s in ['W', 'N1', 'N2', 'N3', 'R']
        }

        # Awakenings: count transitions into W with at least 1 epoch
        awakenings = 0
        for i in range(1, len(stages)):
            if stages[i] == 'W' and stages[i-1] != 'W':
                awakenings += 1

        # Build contiguous periods (start-end in minutes, label)
        periods = []
        if len(stages) > 0:
            start_idx = 0
            for i in range(1, len(stages)):
                if stages[i] != stages[i-1]:
                    periods.append((float(times[start_idx]) / 60.0, float(times[i]) / 60.0, stages[i-1]))
                    start_idx = i
            # last
            periods.append((float(times[start_idx]) / 60.0, float((times[-1] + epoch_len)) / 60.0, stages[-1]))

        # UI: window with summary + table of periods
        top = tk.Toplevel(self.root)
        top.title("Analyse des périodes de sommeil (type SleepEEGpy)")
        top.geometry("1100x900")
        top.transient(self.root)
        top.grab_set()

        container = ttk.Frame(top, padding=10)
        container.pack(fill=tk.BOTH, expand=True)

        summary = (
            f"TIB: {tib:.1f} min | SOL: {sol:.1f} min | TST: {tst_sec/60.0:.1f} min\n"
            f"WASO: {waso_sec/60.0:.1f} min | SE: {se:.1f}% | REM lat.: {rem_lat:.1f} min\n"
            f"W: {stage_durations_min['W']:.1f} | N1: {stage_durations_min['N1']:.1f} | N2: {stage_durations_min['N2']:.1f} | "
            f"N3: {stage_durations_min['N3']:.1f} | R: {stage_durations_min['R']:.1f} | Réveils: {awakenings}"
        )
        ttk.Label(container, text=summary, font=('Helvetica', 10)).pack(anchor='w', pady=(0,10))

        # Boutons d'export
        btns = ttk.Frame(container)
        btns.pack(fill=tk.X, pady=(0,10))
        ttk.Button(btns, text="Exporter CSV", command=lambda: _export_periods_and_metrics_csv(periods, tib, sol, tst_sec, waso_sec, se, rem_lat, stage_durations_min, awakenings)).pack(side=tk.RIGHT)
        ttk.Button(btns, text="Enregistrer Figure", command=lambda: _save_periods_figure(tree)).pack(side=tk.RIGHT, padx=(10,0))

        cols = ("Début (min)", "Fin (min)", "Stade")
        tree = ttk.Treeview(container, columns=cols, show="headings")
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=120, anchor=tk.CENTER)
        tree.pack(fill=tk.BOTH, expand=True)

        for (a, b, s) in periods:
            tree.insert("", tk.END, values=(f"{a:.1f}", f"{b:.1f}", s))

        def _export_periods_and_metrics_csv(periods_list, tib_min, sol_min, tst_seconds, waso_seconds, se_pct, rem_latency_min, stage_dur_min_dict, n_awaken):
            try:
                file_path = filedialog.asksaveasfilename(title="Exporter CSV", defaultextension=".csv",
                                                            filetypes=[("CSV", "*.csv")])
                if not file_path:
                    return
                import csv
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # Metrics
                    writer.writerow(["metric", "value"]) 
                    writer.writerow(["TIB_min", f"{tib_min:.2f}"])
                    writer.writerow(["SOL_min", f"{sol_min:.2f}"])
                    writer.writerow(["TST_min", f"{tst_seconds/60.0:.2f}"])
                    writer.writerow(["WASO_min", f"{waso_seconds/60.0:.2f}"])
                    writer.writerow(["SE_percent", f"{se_pct:.2f}"])
                    writer.writerow(["REM_latency_min", f"{rem_latency_min:.2f}"])
                    for sname, minutes in stage_dur_min_dict.items():
                        writer.writerow([f"Duration_{sname}_min", f"{minutes:.2f}"])
                    writer.writerow(["Awakenings", n_awaken])
                    writer.writerow([])
                    # Periods table
                    writer.writerow(["start_min", "end_min", "stage"]) 
                    for (a, b, s) in periods_list:
                        writer.writerow([f"{a:.2f}", f"{b:.2f}", s])
            except Exception as e:
                messagebox.showerror("Erreur", f"Echec de l'export CSV: {e}")

        def _save_periods_figure(treeview):
            try:
                # Simple export: capture de la fenêtre via matplotlib n'est pas direct; on exporte la liste en figure.
                import matplotlib.pyplot as _plt
                _fig, _ax = _plt.subplots(figsize=(6, len(periods)/6 + 1))
                y = 0
                for (a, b, s) in periods:
                    _ax.plot([a, b], [y, y], lw=6)
                    _ax.text(b + 0.2, y, s)
                    y += 1
                _ax.set_xlabel("Temps (min)")
                _ax.set_title("Périodes de sommeil")
                _ax.set_yticks([])
                _plt.tight_layout()
                file_path = filedialog.asksaveasfilename(title="Enregistrer la figure", defaultextension=".png",
                                                            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
                if file_path:
                    _fig.savefig(file_path, dpi=200, bbox_inches='tight')
                _plt.close(_fig)
            except Exception as e:
                messagebox.showerror("Erreur", f"Echec de l'enregistrement de la figure: {e}")
    except Exception as e:
        messagebox.showerror("Erreur", f"Echec analyse périodes: {e}")


# Ajoute ici d'autres fonctions au besoin relatives au scoring...

