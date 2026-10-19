"""Report building for the auto-vs-manual scoring comparison.

Turns the concordance DataFrame from ``AppController.compare_scoring`` and its
``diagnostics`` attrs into printable sections: the parameters of the automatic
run, the agreement metrics, the dominant confusions and a plain-language
interpretation. Kept free of any Qt import so the dialog and the CSV export
share the same content and the whole thing stays unit-testable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from CESA.scoring_channels import emg_warning
from CESA.scoring_signal_check import amplitude_diagnosis, n3_capability_note
from CESA.scoring_comparison import (
    STAGE_ORDER,
    cohens_kappa,
    confusion_matrix,
    kappa_label,
    macro_f1,
    present_stages,
    stage_metrics,
)

#: Published agreement of each backend against an expert consensus, used to
#: tell an underperforming run from a normal one.
EXPECTED_ACCURACY: Dict[str, Tuple[float, float]] = {
    "yasa": (80.0, 88.0),
    "usleep": (80.0, 88.0),
    "pftsleep": (78.0, 87.0),
    "ml": (70.0, 85.0),
    "ml_hmm": (72.0, 86.0),
    "rules": (60.0, 75.0),
    "rules_hmm": (62.0, 78.0),
    "aasm": (60.0, 75.0),
}

#: Agreement between two human experts on the same night, the practical ceiling
#: for any model.
HUMAN_AGREEMENT = (80.0, 85.0)

_PARAM_LABELS: List[Tuple[str, str]] = [
    ("backend", "Backend"),
    ("date", "Date du calcul"),
    ("duree_calcul_s", "Duree de calcul (s)"),
    ("peripherique", "Peripherique"),
    ("n_epoques", "Epoques produites"),
    ("epoque_s", "Duree d'epoque (s)"),
    ("canal_eeg", "Canal EEG utilise"),
    ("canal_eog", "Canal EOG utilise"),
    ("canal_emg", "Canal EMG utilise"),
    ("canal_ecg", "Canal ECG utilise"),
    ("candidats_eeg", "Candidats EEG (ordre de priorite)"),
    ("candidats_eog", "Candidats EOG (ordre de priorite)"),
    ("candidats_emg", "Candidats EMG (ordre de priorite)"),
    ("sfreq_enregistrement_hz", "Frequence d'echantillonnage source (Hz)"),
    ("sfreq_cible_hz", "Frequence cible (Hz)"),
    ("sfreq_effective_hz", "Frequence effective (Hz)"),
    ("n_canaux_disponibles", "Canaux disponibles dans l'enregistrement"),
    ("seuil_confiance", "Seuil de confiance"),
    ("age", "Age renseigne"),
    ("sexe", "Sexe renseigne"),
    ("correction_n3_aasm", "Correction N3 (critere AASM)"),
    ("canal_n3", "Canal utilise pour le critere N3"),
    ("epoques_promues_n3", "Epoques N2 relevees en N3"),
    ("source_modele", "Source du modele"),
    ("modele_distant", "Modele du service"),
    ("envoi_anonymise", "Envoi anonymise"),
    ("checkpoint", "Checkpoint"),
    ("dossier_poids", "Dossier des poids"),
    ("modele_ml", "Modele ML"),
    ("eog_en_entree", "EOG en entree"),
    ("zscore_par_canal", "Z-score par canal"),
    ("confiance_moyenne", "Confiance moyenne"),
    ("confiance_mediane", "Confiance mediane"),
    ("epoques_sous_seuil", "Epoques sous le seuil de confiance"),
    ("pct_sous_seuil", "Part des epoques sous le seuil (%)"),
]


#: Measured on the EEG channel actually used, to check the signal can carry the
#: slow waves that define N3.
_SIGNAL_LABELS: List[Tuple[str, str]] = [
    ("canal_mesure", "Signal mesure sur"),
    ("amplitude_p2p_uv_mediane", "Amplitude crete a crete mediane (uV)"),
    ("amplitude_p2p_uv_p95", "Amplitude crete a crete p95 (uV)"),
    ("onde_lente_p2p_uv_mediane", "Ondes lentes 0.5-2 Hz, p2p median (uV)"),
    ("onde_lente_p2p_uv_p95", "Ondes lentes 0.5-2 Hz, p2p p95 (uV)"),
    ("pct_epoques_criteres_n3", "Epoques remplissant le critere AASM du N3 (%)"),
]


def _fmt_value(value: Any) -> str:
    if value is None or value == "":
        return "non renseigne"
    if isinstance(value, bool):
        return "oui" if value else "non"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) if value else "aucun"
    return str(value)


def agreement_summary(merged: pd.DataFrame) -> Dict[str, Any]:
    """Global agreement figures for the compared epochs."""
    n_total = int(len(merged))
    n_match = int(merged["match"].sum()) if n_total else 0
    stages = present_stages(merged)
    metrics = stage_metrics(merged, stages) if n_total else {}
    return {
        "n_total": n_total,
        "n_match": n_match,
        "n_mismatch": n_total - n_match,
        "accuracy": (n_match / n_total * 100.0) if n_total else 0.0,
        "kappa": cohens_kappa(merged["auto"], merged["manual"]) if n_total else 0.0,
        "macro_f1": macro_f1(metrics) if metrics else 0.0,
    }


def top_confusions(merged: pd.DataFrame, limit: int = 5) -> List[Dict[str, Any]]:
    """Most frequent disagreements, described as manual stage -> auto stage."""
    errors = merged[~merged["match"].astype(bool)]
    if errors.empty:
        return []
    n_err = len(errors)
    n_total = len(merged) or 1
    counts = errors.groupby(["manual", "auto"]).size().sort_values(ascending=False)
    out: List[Dict[str, Any]] = []
    for (manual, auto), count in list(counts.items())[:limit]:
        out.append({
            "manual": str(manual),
            "auto": str(auto),
            "n": int(count),
            "pct_errors": count / n_err * 100.0,
            "pct_total": count / n_total * 100.0,
        })
    return out


def merged_deep_sleep_accuracy(merged: pd.DataFrame) -> float:
    """Agreement obtained when N2 and N3 are treated as a single stage.

    A large gap with the raw accuracy means the model reads the night correctly
    but places the N2/N3 boundary differently, which is a far milder problem
    than a genuine misreading of the sleep architecture.
    """
    if merged.empty:
        return 0.0
    auto = merged["auto"].replace({"N3": "N2"})
    manual = merged["manual"].replace({"N3": "N2"})
    return float((auto == manual).mean() * 100.0)


def interpretation_notes(
    merged: pd.DataFrame,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Plain-language reading of the result, ordered by importance."""
    diag = dict(diagnostics or {})
    params = dict(diag.get("auto_params") or {})
    notes: List[str] = []
    if merged.empty:
        return ["Aucune epoque comparable."]

    summary = agreement_summary(merged)
    acc = summary["accuracy"]
    backend = str(diag.get("auto_method") or params.get("backend") or "").lower()
    expected = EXPECTED_ACCURACY.get(backend)

    if expected:
        low, high = expected
        if acc < low - 10:
            notes.append(
                f"Concordance de {acc:.1f}% tres inferieure a l'attendu pour "
                f"{backend} ({low:.0f}-{high:.0f}% face a un consensus d'experts) : "
                "un probleme de signal ou de reglage est probable, ce n'est pas "
                "seulement une divergence de jugement."
            )
        elif acc < low:
            notes.append(
                f"Concordance de {acc:.1f}%, en dessous de l'attendu pour {backend} "
                f"({low:.0f}-{high:.0f}%) : resultat exploitable mais a relire."
            )
        else:
            notes.append(
                f"Concordance de {acc:.1f}%, conforme a l'attendu pour {backend} "
                f"({low:.0f}-{high:.0f}%). Pour reference, deux experts humains "
                f"s'accordent sur {HUMAN_AGREEMENT[0]:.0f}-{HUMAN_AGREEMENT[1]:.0f}% "
                "des epoques."
            )

    signal = dict(params.get("signal_eeg") or {})
    amplitude_note = amplitude_diagnosis(signal)
    if amplitude_note:
        notes.append(amplitude_note)

    distribution = params.get("repartition_stades") or {}
    n3_note = n3_capability_note(
        signal,
        predicted_n3_pct=(distribution.get("N3") or {}).get("pct", 0.0),
        manual_n3_pct=(
            float((merged["manual"] == "N3").mean() * 100.0)
            if len(merged) else None
        ),
    )
    if n3_note:
        notes.append(n3_note)

    never = list(diag.get("never_predicted") or [])
    if never:
        supports = {
            stage: int((merged["manual"] == stage).sum()) for stage in never
        }
        detail = ", ".join(
            f"{stage} ({supports[stage]} epoques scorees a la main)"
            for stage in never
        )
        notes.append(
            f"Le modele n'a jamais predit : {detail}. Un stade totalement absent "
            "de la sortie ne s'explique pas par un desaccord de frontiere : "
            "verifiez l'amplitude du signal (unites V/uV, calibration) et la "
            "derivation EEG utilisee, car un stade comme N3 est defini par un "
            "critere d'amplitude (ondes lentes > 75 uV)."
        )

    confusions = top_confusions(merged, limit=3)
    for conf in confusions:
        if conf["pct_errors"] >= 25.0:
            notes.append(
                f"Confusion dominante : {conf['n']} epoques scorees {conf['manual']} "
                f"a la main sont classees {conf['auto']} par le modele, soit "
                f"{conf['pct_errors']:.0f}% de tous les desaccords."
            )

    deep = merged_deep_sleep_accuracy(merged)
    if deep - acc >= 5.0:
        notes.append(
            f"En fusionnant N2 et N3 en un seul stade de sommeil lent, la "
            f"concordance passe a {deep:.1f}% : l'essentiel de l'ecart vient du "
            "placement de la frontiere N2/N3, pas de la structure de la nuit."
        )

    if params.get("pct_sous_seuil") is not None and params["pct_sous_seuil"] >= 20:
        notes.append(
            f"{params['pct_sous_seuil']:.0f}% des epoques sont sous le seuil de "
            f"confiance ({params.get('seuil_confiance')}) : le modele est lui-meme "
            "peu sur de son resultat, ce qui accompagne souvent un signal degrade."
        )

    emg = params.get("canal_emg")
    emg_issue = emg_warning(emg)
    if emg_issue:
        notes.append(emg_issue)
    if backend in ("yasa", "usleep", "pftsleep"):
        rem_as_wake = sum(
            c["n"] for c in top_confusions(merged, limit=25)
            if c["manual"] == "R" and c["auto"] == "W"
        )
        if rem_as_wake >= 20 and (not emg or emg_issue):
            reason = (
                "aucun canal EMG n'a ete utilise"
                if not emg
                else f"le canal EMG utilise ('{emg}') n'est pas un EMG mentonnier"
            )
            notes.append(
                f"{rem_as_wake} epoques de REM sont classees eveil, et {reason}. "
                "L'atonie mentonniere est le critere qui separe le REM de l'eveil : "
                "sans elle, la distinction repose sur l'EEG et l'EOG seuls et ce "
                "type d'erreur est attendu. Un EMG mentonnier (Chin1-Chin2, "
                "Menton) corrigerait ce point ; un EMG de jambe, non."
            )

    if backend == "yasa" and acc < 80.0 and not params.get("age") and "age" in params:
        notes.append(
            "Ni l'age ni le sexe du sujet ne sont renseignes. YASA exploite ces "
            "metadonnees pour ajuster ses predictions ; les fournir est gratuit "
            "et ameliore generalement la concordance de quelques points."
        )

    kappa = summary["kappa"]
    notes.append(
        f"Kappa de Cohen {kappa:.3f} ({kappa_label(kappa)}), qui corrige la "
        "concordance du hasard et reste la mesure a citer."
    )
    return notes


def params_rows(diagnostics: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Label / value pairs describing the automatic run."""
    diag = dict(diagnostics or {})
    params = dict(diag.get("auto_params") or {})
    if not params:
        return []
    rows: List[Tuple[str, str]] = []
    for key, label in _PARAM_LABELS:
        if key in params:
            rows.append((label, _fmt_value(params[key])))
    distribution = params.get("repartition_stades") or {}
    if distribution:
        ordered = [s for s in STAGE_ORDER if s in distribution]
        ordered += [s for s in distribution if s not in ordered]
        rows.append((
            "Repartition predite",
            ", ".join(
                f"{stage} {distribution[stage]['n']} ({distribution[stage]['pct']}%)"
                for stage in ordered
            ),
        ))
    signal = params.get("signal_eeg") or {}
    for key, label in _SIGNAL_LABELS:
        if key in signal:
            rows.append((label, _fmt_value(signal[key])))
    return rows


def alignment_rows(diagnostics: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Label / value pairs describing how the two hypnograms were aligned."""
    diag = dict(diagnostics or {})
    if not diag:
        return []
    rows = [
        ("Backend automatique", _fmt_value(diag.get("auto_method"))),
        ("Source du scoring manuel", _fmt_value(diag.get("manual_source"))),
        ("Epoques du scoring auto", _fmt_value(diag.get("n_auto"))),
        ("Epoques du scoring manuel", _fmt_value(diag.get("n_manual"))),
        ("Epoque auto (s)", _fmt_value(diag.get("auto_epoch_seconds"))),
        ("Epoque manuel (s)", _fmt_value(diag.get("manual_epoch_seconds"))),
        ("Grille de comparaison (s)", _fmt_value(diag.get("comparison_epoch_seconds"))),
        ("Epoques communes", _fmt_value(diag.get("n_common"))),
        ("Epoques comparees", _fmt_value(diag.get("n_compared"))),
        ("Epoques U exclues", _fmt_value(diag.get("n_excluded_u"))),
        ("Epoques hors fenetre de nuit exclues", _fmt_value(diag.get("n_excluded_night"))),
        ("Exclusion des epoques non scorees", _fmt_value(diag.get("exclude_unscored"))),
    ]
    if diag.get("identical") or diag.get("same_object"):
        rows.append(("Alerte", "hypnogrammes identiques : comparaison non valide"))
    if diag.get("auto_is_fallback"):
        rows.append(("Alerte", "scoring actif utilise comme cote auto (repli)"))
    return rows


def write_report_csv(
    path: str,
    merged: pd.DataFrame,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> None:
    """Write the full comparison report as a sectioned CSV.

    Sections are separated by blank lines and introduced by a ``#`` title so the
    file stays readable in a spreadsheet while the per-epoch detail remains a
    plain table at the end.
    """
    diag = dict(diagnostics or {})
    summary = agreement_summary(merged)
    stages = present_stages(merged)
    metrics = stage_metrics(merged, stages) if len(merged) else {}
    lines: List[str] = []

    def section(title: str) -> None:
        if lines:
            lines.append("")
        lines.append(f"# {title}")

    def row(*cells: Any) -> None:
        out = []
        for cell in cells:
            text = "" if cell is None else str(cell)
            if any(ch in text for ch in (',', '"', '\n')):
                text = '"' + text.replace('"', '""') + '"'
            out.append(text)
        lines.append(",".join(out))

    section("Rapport de comparaison scoring automatique vs manuel")
    row("Genere le", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))

    section("Concordance globale")
    row("Indicateur", "Valeur")
    row("Epoques comparees", summary["n_total"])
    row("Epoques concordantes", summary["n_match"])
    row("Epoques discordantes", summary["n_mismatch"])
    row("Concordance (%)", f"{summary['accuracy']:.2f}")
    row("Kappa de Cohen", f"{summary['kappa']:.4f}")
    row("Interpretation du kappa", kappa_label(summary["kappa"]))
    row("F1 macro", f"{summary['macro_f1']:.4f}")
    row("Concordance N2+N3 fusionnes (%)", f"{merged_deep_sleep_accuracy(merged):.2f}")

    params = params_rows(diag)
    if params:
        section("Parametres du scoring automatique")
        row("Parametre", "Valeur")
        for label, value in params:
            row(label, value)

    align = alignment_rows(diag)
    if align:
        section("Alignement des deux scorings")
        row("Parametre", "Valeur")
        for label, value in align:
            row(label, value)

    if metrics:
        section("Metriques par stade (reference = scoring manuel)")
        row("Stade", "Support manuel", "Predites auto", "Sensibilite %",
            "Precision %", "Specificite %", "F1 %", "VP", "FP", "FN")
        for stage in stages:
            m = metrics[stage]

            def pct(value: float) -> str:
                return "" if value != value else f"{value * 100:.1f}"

            row(stage, int(m["support"]), int(m["n_auto"]), pct(m["recall"]),
                pct(m["precision"]), pct(m["specificity"]), pct(m["f1"]),
                int(m["tp"]), int(m["fp"]), int(m["fn"]))

    if stages:
        section("Matrice de confusion (lignes = auto, colonnes = manuel)")
        cm = confusion_matrix(merged, stages)
        row("auto \\ manuel", *stages)
        for i, stage in enumerate(stages):
            row(stage, *[int(v) for v in cm[i]])

    confusions = top_confusions(merged, limit=10)
    if confusions:
        section("Principales confusions")
        row("Stade manuel", "Classe par le modele", "Epoques",
            "% des desaccords", "% du total")
        for conf in confusions:
            row(conf["manual"], conf["auto"], conf["n"],
                f"{conf['pct_errors']:.1f}", f"{conf['pct_total']:.1f}")

    notes = interpretation_notes(merged, diag)
    if notes:
        section("Interpretation")
        for note in notes:
            row(note)

    section("Detail par epoque")
    columns = list(merged.columns)
    row(*columns)
    for record in merged.itertuples(index=False):
        row(*record)

    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        handle.write("\n".join(lines) + "\n")
