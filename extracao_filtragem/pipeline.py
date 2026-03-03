#!/usr/bin/env python3
"""
Pipeline LDBC SNB Fitness - Extração e filtragem de conteúdo fitness do LDBC SNB Interactive v1.
Gera dataset de interações (likes, criação, reply) para treinamento de IA.
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import duckdb


# --- Configuração (paths relativos ao script) ---
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
EXTRACAO_DIR = _SCRIPT_DIR
LDBC_SNB_DIR = EXTRACAO_DIR / "ldbc_snb"
OUTPUT_DIR = EXTRACAO_DIR / "output"
TREINAMENTO_DIR = _PROJECT_ROOT / "treinamento"
DATASET_DIR = EXTRACAO_DIR / "dataset"
DEFAULT_DATASET = DATASET_DIR / "social_network-sf0.1-CsvBasic-LongDateFormatter.tar.zst"

TAGCLASS_FITNESS_KEYWORDS = [
    "sports", "health", "fitness", "running", "exercise", "gym", "athletic",
]
TAG_FITNESS_KEYWORDS = [
    "treino", "academia", "corrida", "run", "running", "gym", "fitness",
    "workout", "marathon", "hiit", "crossfit", "musculação", "musculacao",
    "exercise", "sport", "sports", "treino", "correr", "jogging", "yoga",
    "pilates", "bodybuilding", "cardio", "weight", "lifting",
]


def log(msg: str) -> None:
    print(f"[Pipeline] {msg}", flush=True)


def q(col: str) -> str:
    """Quota nome de coluna para SQL (ex: Post.id -> "Post.id")."""
    return f'"{col}"' if col and "." in col else (col or "")



def ensure_dirs() -> None:
    """Cria pastas extracao_filtragem/dataset, ldbc_snb, output e treinamento."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    LDBC_SNB_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TREINAMENTO_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Pastas criadas: {DATASET_DIR}, {OUTPUT_DIR}, {TREINAMENTO_DIR}")


def extract_archive(dataset_path: Path) -> Path:
    """Extrai .tar.zst com zstd + tar. Retorna o diretório base dos CSVs."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_path}")

    log(f"Extraindo {dataset_path} para {LDBC_SNB_DIR}...")
    zstd_cmd = ["zstd", "-d", "-c", str(dataset_path)]
    tar_cmd = ["tar", "-xf", "-", "-C", str(LDBC_SNB_DIR)]

    try:
        p1 = subprocess.Popen(zstd_cmd, stdout=subprocess.PIPE)
        p2 = subprocess.Popen(tar_cmd, stdin=p1.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        p1.stdout.close()
        _, err = p2.communicate()
        if p2.returncode != 0 and err:
            log(f"Aviso tar: {err.decode(errors='ignore')}")
    except FileNotFoundError as e:
        raise RuntimeError(
            "zstd ou tar não encontrado. Instale zstd (ex: choco install zstd) e tar (Windows 10+ inclui)."
        ) from e

    log("Extração concluída.")
    return LDBC_SNB_DIR


def discover_csv_files(base: Path) -> dict[str, list[Path]]:
    """Descobre CSVs por padrão de nome. Apenas arquivos de entidade (não relação)."""
    import re
    all_csvs = list(base.rglob("*.csv"))
    mapping: dict[str, list[Path]] = {
        "tag": [],
        "tagclass": [],
        "post": [],
        "comment": [],
        "post_hascreator_person": [],
        "comment_hascreator_person": [],
        "post_hastag_tag": [],
        "comment_hastag_tag": [],
        "comment_replyof_post": [],
        "comment_replyof_comment": [],
        "person_likes_post": [],
        "person_likes_comment": [],
        "person_hasinterest_tag": [],
        "person_knows_person": [],
        "forum_hastag_tag": [],
        "forum_containerof_post": [],
    }

    for p in all_csvs:
        name = p.name.lower()
        if re.match(r"^tag_\d+_\d+\.csv$", name) or name == "tag.csv":
            mapping["tag"].append(p)
        elif re.match(r"^tagclass_\d+_\d+\.csv$", name) or name == "tagclass.csv":
            mapping["tagclass"].append(p)
        elif re.match(r"^post_\d+_\d+\.csv$", name) or name == "post.csv":
            mapping["post"].append(p)
        elif re.match(r"^comment_\d+_\d+\.csv$", name) or name == "comment.csv":
            mapping["comment"].append(p)
        elif "post_hascreator_person" in name or "post_has_creator" in name:
            mapping["post_hascreator_person"].append(p)
        elif "comment_hascreator_person" in name or "comment_has_creator" in name:
            mapping["comment_hascreator_person"].append(p)
        elif "post_hastag_tag" in name or "post_has_tag" in name:
            mapping["post_hastag_tag"].append(p)
        elif "comment_hastag_tag" in name or "comment_has_tag" in name:
            mapping["comment_hastag_tag"].append(p)
        elif "comment_replyof_post" in name:
            mapping["comment_replyof_post"].append(p)
        elif "comment_replyof_comment" in name:
            mapping["comment_replyof_comment"].append(p)
        elif "person_likes_post" in name:
            mapping["person_likes_post"].append(p)
        elif "person_likes_comment" in name:
            mapping["person_likes_comment"].append(p)
        elif "person_hasinterest_tag" in name or "person_has_interest" in name:
            mapping["person_hasinterest_tag"].append(p)
        elif "person_knows_person" in name:
            mapping["person_knows_person"].append(p)
        elif "forum_hastag_tag" in name or "forum_has_tag" in name:
            mapping["forum_hastag_tag"].append(p)
        elif "forum_containerof_post" in name or "forum_container" in name:
            mapping["forum_containerof_post"].append(p)

    for k, v in mapping.items():
        mapping[k] = sorted(v)
    return mapping


def detect_csv_format(csv_path: Path) -> tuple[str, bool]:
    """Detecta delimitador e se tem header. Retorna (delim, has_header)."""
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        sample = "".join(f.readline() for _ in range(3))
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="|,\t")
        has_header = csv.Sniffer().has_header(sample)
        return (dialect.delimiter, has_header)
    except csv.Error:
        return ("|", False)


def load_tables_flexible(con: duckdb.DuckDBPyConnection, mapping: dict) -> None:
    """Carrega cada tipo de CSV em tabela DuckDB, com fallback de detecção."""
    delim = "|"
    header = False

    for key, paths in mapping.items():
        if not paths:
            continue
        tbl = key
        first = paths[0]
        try:
            d, h = detect_csv_format(first)
            delim, header = d, h
        except Exception:
            pass

        path_strs = [str(p.resolve()).replace("\\", "/") for p in paths]
        paths_sql = "[" + ", ".join("'" + p.replace("'", "''") + "'" for p in path_strs) + "]"

        try:
            sql = f"""
                CREATE OR REPLACE TABLE {tbl} AS
                SELECT * FROM read_csv_auto(
                    {paths_sql},
                    delim='{delim}',
                    header={str(header).lower()},
                    ignore_errors=true
                )
            """
            con.execute(sql)
            n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            log(f"  {tbl}: {n} linhas")
        except Exception as e:
            log(f"  Erro {tbl}: {e}")


def get_tag_fitness_ids(con: duckdb.DuckDBPyConnection) -> list[int]:
    """Estratégia A: TagClass fitness + descendentes. Fallback B: Tags por nome."""
    tag_ids: set[int] = set()

    # Estratégia A: TagClass
    try:
        cols = [c[0] for c in con.execute("DESCRIBE tagclass").fetchall()]
        id_col = "id" if "id" in [c.lower() for c in cols] else cols[0]
        name_col = "name" if "name" in [c.lower() for c in cols] else next((c for c in cols if "name" in c.lower()), cols[1])
        sub_col = next((c for c in cols if "subclass" in c.lower() or "parent" in c.lower()), None)

        tc_conditions = " OR ".join(
            f"LOWER({name_col}) LIKE '%{kw}%'" for kw in TAGCLASS_FITNESS_KEYWORDS
        )
        tc_ids = con.execute(f"""
            SELECT {id_col} FROM tagclass WHERE {tc_conditions}
        """).fetchall()
        tc_ids = {r[0] for r in tc_ids}

        if sub_col and tc_ids:
            # Expandir descendentes
            while True:
                extra = con.execute(f"""
                    SELECT {id_col} FROM tagclass
                    WHERE {sub_col} IN ({','.join(map(str, tc_ids))})
                    AND {id_col} NOT IN ({','.join(map(str, tc_ids))})
                """).fetchall()
                if not extra:
                    break
                tc_ids.update(r[0] for r in extra)

        if tc_ids:
            tag_cols = [c[0] for c in con.execute("DESCRIBE tag").fetchall()]
            tc_fk = next((c for c in tag_cols if "tagclass" in c.lower() or "type" in c.lower()), "hasType_TagClass")
            tag_ids.update(
                r[0] for r in con.execute(
                    f"SELECT id FROM tag WHERE {tc_fk} IN ({','.join(map(str, tc_ids))})"
                ).fetchall()
            )
    except Exception as e:
        log(f"  Estratégia A (TagClass) falhou: {e}")

    # Estratégia B: Tags por nome
    try:
        tag_name_col = "name"
        for kw in TAG_FITNESS_KEYWORDS:
            rows = con.execute(
                f"SELECT id FROM tag WHERE LOWER({tag_name_col}) LIKE '%{kw}%'"
            ).fetchall()
            tag_ids.update(r[0] for r in rows)
    except Exception as e:
        log(f"  Estratégia B (Tag nome) parcial: {e}")

    return list(tag_ids)


def run_pipeline(dataset_path: Path) -> None:
    """Executa o pipeline completo."""
    ensure_dirs()
    base = extract_archive(dataset_path)
    mapping = discover_csv_files(base)

    log("Arquivos descobertos:")
    for k, v in mapping.items():
        if v:
            log(f"  {k}: {len(v)} arquivo(s)")

    # Validar: Tag e TagClass são obrigatórios
    if not mapping["tag"] or not mapping["tagclass"]:
        all_csvs = list(base.rglob("*.csv"))
        sample_names = [p.name for p in all_csvs[:15]] if all_csvs else []
        raise FileNotFoundError(
            "O dataset não contém os arquivos necessários (Tag.csv, TagClass.csv, Post, Comment, etc.).\n"
            f"Arquivos encontrados: {sample_names}...\n\n"
            "O arquivo 'social_network-sf30-numpart-8.tar.zst' contém apenas update streams (forum, person),\n"
            "não o snapshot completo. Use o dataset completo, por exemplo:\n"
            "  social_network-sf30-CsvBasic-LongDateFormatter.tar.zst\n"
            "ou social_network-csv-basic-sf30.tar.zst (SURF/CWI repository)."
        )

    con = duckdb.connect(":memory:")

    # Carregar tabelas
    log("Carregando CSVs no DuckDB...")
    load_tables_flexible(con, mapping)

    # Tags fitness
    log("Aplicando filtro fitness...")
    tag_fitness_ids = get_tag_fitness_ids(con)
    log(f"  Tags fitness: {len(tag_fitness_ids)}")

    if not tag_fitness_ids:
        log("AVISO: Nenhuma tag fitness encontrada. Usando todas as tags para mensagens.")
        tag_fitness_ids = [r[0] for r in con.execute("SELECT id FROM tag").fetchall()]

    if not tag_fitness_ids:
        log("ERRO: Nenhuma tag disponível. Verifique os arquivos Tag e TagClass.")
        sys.exit(1)

    con.execute("CREATE TEMP TABLE tag_fitness_ids AS SELECT unnest(?) AS id", [tag_fitness_ids])

    # Messages fitness (Post + Comment com pelo menos 1 tag fitness)
    message_ids: set[int] = set()
    try:
        for tbl in ["post_hastag_tag", "comment_hastag_tag"]:
            if con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0] == 0:
                continue
            cols = [c[0] for c in con.execute(f"DESCRIBE {tbl}").fetchall()]
            id_col = next((c for c in cols if "post" in c.lower() or "comment" in c.lower()), cols[0])
            hastag_col = next((c for c in cols if "tag" in c.lower() and c != id_col), cols[1])
            rows = con.execute(f"""
                SELECT DISTINCT {q(id_col)} FROM {tbl}
                WHERE {q(hastag_col)} IN (SELECT id FROM tag_fitness_ids)
            """).fetchall()
            message_ids.update(r[0] for r in rows)
    except Exception as e:
        log(f"  Erro ao obter messages fitness: {e}")

    log(f"  Messages fitness: {len(message_ids)}")

    if not message_ids:
        log("AVISO: Nenhum message fitness. Gerando interações vazias.")

    con.execute("CREATE TEMP TABLE message_fitness_ids AS SELECT unnest(?) AS id", [list(message_ids)])

    # Tags por message (para tags_fitness na lista)
    msg_tag_map: dict[int, list[str]] = {}
    try:
        for tbl in ["post_hastag_tag", "comment_hastag_tag"]:
            if con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0] == 0:
                continue
            cols = [c[0] for c in con.execute(f"DESCRIBE {tbl}").fetchall()]
            id_col = next((c for c in cols if "post" in c.lower() or "comment" in c.lower()), cols[0])
            hastag_col = next((c for c in cols if "tag" in c.lower() and c != id_col), cols[1])
            rows = con.execute(f"""
                SELECT m.{q(id_col)}, t.name
                FROM {tbl} m
                JOIN tag t ON t.id = m.{q(hastag_col)}
                WHERE m.{q(hastag_col)} IN (SELECT id FROM tag_fitness_ids)
            """).fetchall()
            for mid, tname in rows:
                msg_tag_map.setdefault(mid, []).append(str(tname))
    except Exception as e:
        log(f"  Erro tags por message: {e}")

    # Interações
    interactions: list[dict] = []

    def add_interaction(user_id: int, msg_id: int, event_type: str, ts: str) -> None:
        if msg_id not in message_ids:
            return
        tags = msg_tag_map.get(msg_id, [])
        interactions.append({
            "user_id": user_id,
            "message_id": msg_id,
            "event_type": event_type,
            "timestamp": str(ts),
            "tags_fitness": tags,
        })

    # Likes
    try:
        for tbl in ["person_likes_post", "person_likes_comment"]:
            if con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0] == 0:
                continue
            cols = [c[0] for c in con.execute(f"DESCRIBE {tbl}").fetchall()]
            uid = next((c for c in cols if "person" in c.lower() and "id" in c.lower()), cols[0])
            mid = next((c for c in cols if "post" in c.lower() or "comment" in c.lower()), cols[1])
            ts_col = next((c for c in cols if "date" in c.lower() or "creation" in c.lower()), cols[-1])
            rows = con.execute(f"""
                SELECT DISTINCT l.{q(uid)}, l.{q(mid)}, l.{q(ts_col)}
                FROM {tbl} l
                WHERE l.{q(mid)} IN (SELECT id FROM message_fitness_ids)
            """).fetchall()
            for uid_val, mid_val, ts_val in rows:
                add_interaction(int(uid_val), int(mid_val), "like", str(ts_val))
    except Exception as e:
        log(f"  Erro likes: {e}")

    # Create (hasCreator) - suporta CsvBasic (tabela separada) ou merged (coluna no post/comment)
    try:
        for msg_tbl, creator_tbl in [("post", "post_hascreator_person"), ("comment", "comment_hascreator_person")]:
            if con.execute(f"SELECT COUNT(*) FROM {msg_tbl}").fetchone()[0] == 0:
                continue
            msg_cols = [c[0] for c in con.execute(f"DESCRIBE {msg_tbl}").fetchall()]
            msg_id_col = next((c for c in msg_cols if c.lower() == "id" or c.endswith(".id")), msg_cols[0])
            msg_ts_col = next((c for c in msg_cols if "date" in c.lower() or "creation" in c.lower()), msg_cols[0])

            if con.execute(f"SELECT COUNT(*) FROM {creator_tbl}").fetchone()[0] > 0:
                cr_cols = [c[0] for c in con.execute(f"DESCRIBE {creator_tbl}").fetchall()]
                msg_fk = next((c for c in cr_cols if "post" in c.lower() or "comment" in c.lower()), cr_cols[0])
                uid_col = next((c for c in cr_cols if "person" in c.lower()), cr_cols[1])
                rows = con.execute(f"""
                    SELECT hc.{q(uid_col)}, hc.{q(msg_fk)}, m.{q(msg_ts_col)}
                    FROM {creator_tbl} hc
                    JOIN {msg_tbl} m ON m.{q(msg_id_col)} = hc.{q(msg_fk)}
                    WHERE hc.{q(msg_fk)} IN (SELECT id FROM message_fitness_ids)
                """).fetchall()
            else:
                uid = next((c for c in msg_cols if "creator" in c.lower() or "person" in c.lower()), None)
                if not uid:
                    continue
                rows = con.execute(f"""
                    SELECT p.{q(uid)}, p.{q(msg_id_col)}, p.{q(msg_ts_col)}
                    FROM {msg_tbl} p
                    WHERE p.{q(msg_id_col)} IN (SELECT id FROM message_fitness_ids)
                """).fetchall()
            for uid_val, mid_val, ts_val in rows:
                add_interaction(int(uid_val), int(mid_val), "create", str(ts_val))
    except Exception as e:
        log(f"  Erro create: {e}")

    # Reply (comment é reply - CsvBasic usa tabelas comment_replyOf_post/comment)
    try:
        reply_ids: list[int] = []
        for tbl in ["comment_replyof_post", "comment_replyof_comment"]:
            try:
                if con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0] == 0:
                    continue
                cols = [c[0] for c in con.execute(f"DESCRIBE {tbl}").fetchall()]
                cid_col = next((c for c in cols if "comment" in c.lower()), cols[0])
                reply_ids.extend(r[0] for r in con.execute(f"SELECT {q(cid_col)} FROM {tbl}").fetchall())
            except Exception:
                pass
        reply_ids = list(set(reply_ids) & message_ids)
        if reply_ids and con.execute("SELECT COUNT(*) FROM comment_hascreator_person").fetchone()[0] > 0:
            con.execute("CREATE TEMP TABLE reply_comment_ids AS SELECT unnest(?) AS id", [reply_ids])
            cr_cols = [c[0] for c in con.execute("DESCRIBE comment_hascreator_person").fetchall()]
            msg_fk = next((c for c in cr_cols if "comment" in c.lower()), cr_cols[0])
            uid_col = next((c for c in cr_cols if "person" in c.lower()), cr_cols[1])
            comm_cols = [c[0] for c in con.execute("DESCRIBE comment").fetchall()]
            cid = next((c for c in comm_cols if c.lower() == "id" or "comment" in c.lower()), comm_cols[0])
            ts_col = next((c for c in comm_cols if "date" in c.lower()), comm_cols[0])
            rows = con.execute(f"""
                SELECT hc.{q(uid_col)}, hc.{q(msg_fk)}, c.{q(ts_col)}
                FROM comment_hascreator_person hc
                JOIN comment c ON c.{q(cid)} = hc.{q(msg_fk)}
                WHERE hc.{q(msg_fk)} IN (SELECT id FROM reply_comment_ids)
            """).fetchall()
            for uid_val, mid_val, ts_val in rows:
                add_interaction(int(uid_val), int(mid_val), "reply", str(ts_val))
    except Exception as e:
        log(f"  Erro reply: {e}")

    log(f"  Interações: {len(interactions)}")

    # Exportar Parquet
    import pyarrow as pa
    import pyarrow.parquet as pq

    # interactions_fitness.parquet
    if interactions:
        arr = pa.array([tuple(x["tags_fitness"]) for x in interactions], type=pa.list_(pa.string()))
        tbl_inter = pa.table({
            "user_id": [x["user_id"] for x in interactions],
            "message_id": [x["message_id"] for x in interactions],
            "event_type": [x["event_type"] for x in interactions],
            "timestamp": [x["timestamp"] for x in interactions],
            "tags_fitness": arr,
        })
    else:
        tbl_inter = pa.table({
            "user_id": pa.array([], type=pa.int64()),
            "message_id": pa.array([], type=pa.int64()),
            "event_type": pa.array([], type=pa.string()),
            "timestamp": pa.array([], type=pa.string()),
            "tags_fitness": pa.array([], type=pa.list_(pa.string())),
        })
    pq.write_table(tbl_inter, OUTPUT_DIR / "interactions_fitness.parquet")
    log(f"Exportado: {OUTPUT_DIR / 'interactions_fitness.parquet'}")

    # messages_fitness.parquet — enriquecido com creation_date, content_length, language, forum_id
    msg_rows: list[dict] = []
    try:
        # Carregar forum_containerof_post para lookup post -> forum
        forum_of_post: dict[int, int] = {}
        if mapping.get("forum_containerof_post"):
            try:
                fc_cols = [c[0] for c in con.execute("DESCRIBE forum_containerof_post").fetchall()]
                forum_col = next((c for c in fc_cols if "forum" in c.lower()), fc_cols[0])
                post_col = next((c for c in fc_cols if "post" in c.lower()), fc_cols[1])
                for row in con.execute(
                    f'SELECT {q(forum_col)}, {q(post_col)} FROM forum_containerof_post'
                ).fetchall():
                    forum_of_post[int(row[1])] = int(row[0])
            except Exception as e:
                log(f"  Aviso forum_containerof_post: {e}")

        # Posts fitness
        post_cols = [c[0] for c in con.execute("DESCRIBE post").fetchall()]
        pid_col = next((c for c in post_cols if c.lower() == "id"), post_cols[0])
        pts_col = next((c for c in post_cols if "date" in c.lower() or "creation" in c.lower()), post_cols[0])
        plen_col = next((c for c in post_cols if "length" in c.lower()), None)
        plang_col = next((c for c in post_cols if "language" in c.lower() or "lang" in c.lower()), None)
        p_select_parts = [q(pid_col), q(pts_col)]
        p_idx = {"id": 0, "ts": 1, "len": None, "lang": None}
        idx = 2
        if plen_col:
            p_select_parts.append(q(plen_col))
            p_idx["len"] = idx; idx += 1
        if plang_col:
            p_select_parts.append(q(plang_col))
            p_idx["lang"] = idx
        p_select = ", ".join(p_select_parts)
        for row in con.execute(
            f"SELECT {p_select} FROM post WHERE {q(pid_col)} IN (SELECT id FROM message_fitness_ids)"
        ).fetchall():
            mid = int(row[0])
            msg_rows.append({
                "message_id": mid,
                "message_type": "post",
                "creation_date": str(row[1]),
                "content_length": int(row[p_idx["len"]]) if p_idx["len"] is not None else None,
                "language": str(row[p_idx["lang"]]) if p_idx["lang"] is not None else None,
                "forum_id": forum_of_post.get(mid),
                "tags_fitness": msg_tag_map.get(mid, []),
            })

        # Comments fitness
        com_cols = [c[0] for c in con.execute("DESCRIBE comment").fetchall()]
        cid_col = next((c for c in com_cols if c.lower() == "id"), com_cols[0])
        cts_col = next((c for c in com_cols if "date" in c.lower() or "creation" in c.lower()), com_cols[0])
        clen_col = next((c for c in com_cols if "length" in c.lower()), None)
        c_select_parts = [q(cid_col), q(cts_col)]
        c_len_idx: int | None = None
        if clen_col:
            c_select_parts.append(q(clen_col))
            c_len_idx = 2
        c_select = ", ".join(c_select_parts)
        for row in con.execute(
            f"SELECT {c_select} FROM comment WHERE {q(cid_col)} IN (SELECT id FROM message_fitness_ids)"
        ).fetchall():
            mid = int(row[0])
            msg_rows.append({
                "message_id": mid,
                "message_type": "comment",
                "creation_date": str(row[1]),
                "content_length": int(row[c_len_idx]) if c_len_idx is not None else None,
                "language": None,
                "forum_id": None,
                "tags_fitness": msg_tag_map.get(mid, []),
            })
    except Exception as e:
        log(f"  Erro ao enriquecer messages: {e}")
        # fallback simples
        msg_ids_fb = list(message_ids)
        for mid in msg_ids_fb:
            msg_rows.append({
                "message_id": mid, "message_type": "unknown",
                "creation_date": None, "content_length": None,
                "language": None, "forum_id": None,
                "tags_fitness": msg_tag_map.get(mid, []),
            })

    if msg_rows:
        tbl_msg = pa.table({
            "message_id": [r["message_id"] for r in msg_rows],
            "message_type": [r["message_type"] for r in msg_rows],
            "creation_date": [r["creation_date"] for r in msg_rows],
            "content_length": pa.array([r["content_length"] for r in msg_rows], type=pa.int64()),
            "language": [r["language"] for r in msg_rows],
            "forum_id": pa.array([r["forum_id"] for r in msg_rows], type=pa.int64()),
            "tags_fitness": pa.array([r["tags_fitness"] for r in msg_rows], type=pa.list_(pa.string())),
        })
    else:
        tbl_msg = pa.table({
            "message_id": pa.array([], type=pa.int64()),
            "message_type": pa.array([], type=pa.string()),
            "creation_date": pa.array([], type=pa.string()),
            "content_length": pa.array([], type=pa.int64()),
            "language": pa.array([], type=pa.string()),
            "forum_id": pa.array([], type=pa.int64()),
            "tags_fitness": pa.array([], type=pa.list_(pa.string())),
        })
    pq.write_table(tbl_msg, OUTPUT_DIR / "messages_fitness.parquet")
    log(f"Exportado: {OUTPUT_DIR / 'messages_fitness.parquet'} ({len(msg_rows)} linhas)")

    # tags_fitness.parquet
    tag_rows = con.execute("""
        SELECT id, name FROM tag WHERE id IN (SELECT id FROM tag_fitness_ids)
    """).fetchall()
    if tag_rows:
        tbl_tag = pa.table({
            "tag_id": [r[0] for r in tag_rows],
            "tag_name": [r[1] for r in tag_rows],
        })
    else:
        tbl_tag = pa.table({
            "tag_id": pa.array([], type=pa.int64()),
            "tag_name": pa.array([], type=pa.string()),
        })
    pq.write_table(tbl_tag, OUTPUT_DIR / "tags_fitness.parquet")
    log(f"Exportado: {OUTPUT_DIR / 'tags_fitness.parquet'}")

    # user_interests_fitness.parquet
    interest_rows: list[tuple] = []
    try:
        if mapping.get("person_hasinterest_tag"):
            pi_cols = [c[0] for c in con.execute("DESCRIBE person_hasinterest_tag").fetchall()]
            pi_uid = next((c for c in pi_cols if "person" in c.lower()), pi_cols[0])
            pi_tag = next((c for c in pi_cols if "tag" in c.lower()), pi_cols[1])
            interest_rows = con.execute(f"""
                SELECT pit.{q(pi_uid)}, t.id, t.name
                FROM person_hasinterest_tag pit
                JOIN tag t ON t.id = pit.{q(pi_tag)}
                WHERE pit.{q(pi_tag)} IN (SELECT id FROM tag_fitness_ids)
            """).fetchall()
    except Exception as e:
        log(f"  Aviso user_interests: {e}")

    if interest_rows:
        tbl_interests = pa.table({
            "user_id": [int(r[0]) for r in interest_rows],
            "tag_id": [int(r[1]) for r in interest_rows],
            "tag_name": [str(r[2]) for r in interest_rows],
        })
    else:
        tbl_interests = pa.table({
            "user_id": pa.array([], type=pa.int64()),
            "tag_id": pa.array([], type=pa.int64()),
            "tag_name": pa.array([], type=pa.string()),
        })
    pq.write_table(tbl_interests, OUTPUT_DIR / "user_interests_fitness.parquet")
    log(f"Exportado: {OUTPUT_DIR / 'user_interests_fitness.parquet'} ({len(interest_rows)} linhas)")

    # user_social_graph.parquet — pares onde ao menos 1 participou de interações fitness
    social_rows: list[tuple] = []
    try:
        if mapping.get("person_knows_person") and interactions:
            active_users = list({x["user_id"] for x in interactions})
            con.execute("CREATE TEMP TABLE active_users AS SELECT unnest(?) AS id", [active_users])
            pk_cols = [c[0] for c in con.execute("DESCRIBE person_knows_person").fetchall()]
            pk_p1 = pk_cols[0]
            pk_p2 = pk_cols[1]
            pk_ts = next((c for c in pk_cols if "date" in c.lower() or "creation" in c.lower()), None)
            ts_sel = f", pkp.{q(pk_ts)}" if pk_ts else ""
            social_rows = con.execute(f"""
                SELECT pkp.{q(pk_p1)}, pkp.{q(pk_p2)}{ts_sel}
                FROM person_knows_person pkp
                WHERE pkp.{q(pk_p1)} IN (SELECT id FROM active_users)
                   OR pkp.{q(pk_p2)} IN (SELECT id FROM active_users)
            """).fetchall()
    except Exception as e:
        log(f"  Aviso user_social_graph: {e}")

    if social_rows:
        has_ts = len(social_rows[0]) == 3
        tbl_social = pa.table({
            "user_id": [int(r[0]) for r in social_rows],
            "friend_id": [int(r[1]) for r in social_rows],
            "since": [str(r[2]) if has_ts else None for r in social_rows],
        })
    else:
        tbl_social = pa.table({
            "user_id": pa.array([], type=pa.int64()),
            "friend_id": pa.array([], type=pa.int64()),
            "since": pa.array([], type=pa.string()),
        })
    pq.write_table(tbl_social, OUTPUT_DIR / "user_social_graph.parquet")
    log(f"Exportado: {OUTPUT_DIR / 'user_social_graph.parquet'} ({len(social_rows)} linhas)")

    # tag_cooccurrence.parquet — pares de tags que aparecem juntas nos messages fitness
    cooc_rows: list[tuple] = []
    try:
        # Descobrir colunas de cada tabela _hastag_tag dinamicamente
        union_parts: list[str] = []
        for tbl in ["post_hastag_tag", "comment_hastag_tag"]:
            try:
                if con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0] == 0:
                    continue
                cols = [c[0] for c in con.execute(f"DESCRIBE {tbl}").fetchall()]
                id_c = next((c for c in cols if "post" in c.lower() or "comment" in c.lower()), cols[0])
                tag_c = next((c for c in cols if "tag" in c.lower() and c != id_c), cols[1])
                union_parts.append(
                    f"SELECT {q(id_c)} AS msg_id, {q(tag_c)} AS tag_id FROM {tbl}"
                )
            except Exception:
                pass

        if union_parts:
            union_sql = " UNION ALL ".join(union_parts)
            cooc_rows = con.execute(f"""
                WITH base AS (
                    SELECT ht.msg_id, ht.tag_id
                    FROM ({union_sql}) ht
                    JOIN message_fitness_ids mfi ON mfi.id = ht.msg_id
                    JOIN tag_fitness_ids tfi ON tfi.id = ht.tag_id
                )
                SELECT
                    ta.name AS tag_a,
                    tb.name AS tag_b,
                    COUNT(*) AS cooccurrences
                FROM base a
                JOIN base b ON a.msg_id = b.msg_id AND a.tag_id < b.tag_id
                JOIN tag ta ON ta.id = a.tag_id
                JOIN tag tb ON tb.id = b.tag_id
                GROUP BY ta.name, tb.name
                ORDER BY cooccurrences DESC
            """).fetchall()
    except Exception as e:
        log(f"  Aviso tag_cooccurrence: {e}")
        cooc_rows = []

    if cooc_rows:
        tbl_cooc = pa.table({
            "tag_a": [str(r[0]) for r in cooc_rows],
            "tag_b": [str(r[1]) for r in cooc_rows],
            "cooccurrences": [int(r[2]) for r in cooc_rows],
        })
    else:
        tbl_cooc = pa.table({
            "tag_a": pa.array([], type=pa.string()),
            "tag_b": pa.array([], type=pa.string()),
            "cooccurrences": pa.array([], type=pa.int64()),
        })
    pq.write_table(tbl_cooc, OUTPUT_DIR / "tag_cooccurrence.parquet")
    log(f"Exportado: {OUTPUT_DIR / 'tag_cooccurrence.parquet'} ({len(cooc_rows)} pares)")

    con.close()

    log("Pipeline concluído.")
    log(f"Resumo:")
    log(f"  Tags fitness:            {len(tag_fitness_ids)}")
    log(f"  Messages fitness:        {len(message_ids)}")
    log(f"  Interações:              {len(interactions)}")
    log(f"  Interesses de usuário:   {len(interest_rows)}")
    log(f"  Arestas grafo social:    {len(social_rows)}")
    log(f"  Pares co-ocorrência:     {len(cooc_rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline LDBC SNB Fitness")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(os.environ.get("LDBC_DATASET_PATH", str(DEFAULT_DATASET))),
        help="Caminho do arquivo .tar.zst",
    )
    args = parser.parse_args()
    run_pipeline(args.dataset_path)


if __name__ == "__main__":
    main()
