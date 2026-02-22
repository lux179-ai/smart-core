import json
import random
import uuid
import os
import time
from typing import List, Dict, Optional, Any, Tuple
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import redis

# --- CONFIGURAZIONE ---
# Legge le variabili d'ambiente (settate dal Dockerfile o dal sistema)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
SESSION_TTL = 1800  # 30 minuti di vita per ogni sessione di gioco

app = FastAPI(title="SMART RIE - Relational Inference Engine v1.0")

# --- REDIS CONNECTION MANAGER ---
# --- COPIA E INCOLLA QUESTO AL POSTO DEL BLOCCO TRY/EXCEPT ---
USE_REDIS = False
memory_store = {}
print("⚠️ [SYSTEM] Redis bypassato correttamente. Utilizzo memoria locale.")

# --- MODELLI DATI (Pydantic) ---

class SpatialData(BaseModel):
    x: float
    y: float
    is_highest_node: bool = False
    clustering_weight: float = 0.0

class EdgeModel(BaseModel):
    source: str
    relation: str
    target: str

class ValidationRequest(BaseModel):
    session_id: str
    user_edge: EdgeModel
    thinking_time_ms: int
    spatial_data: Optional[SpatialData] = None

# --- COSTANTI E MAPPE RELAZIONALI ---

INVERSE_RELATIONS = {
    "EQUALS": "EQUALS", "DISTINCT": "DISTINCT",
    "GREATER": "LESS", "LESS": "GREATER",
    "CONTAINS": "INSIDE", "INSIDE": "CONTAINS",
    "BEFORE": "AFTER", "AFTER": "BEFORE",
    "OPPOSITE": "OPPOSITE",
    "DURING": "CONTAINS_TIME", "CONTAINS_TIME": "DURING"
}

RELATION_TEXT = {
    "EQUALS": "è identico a", "DISTINCT": "è diverso da",
    "GREATER": "è maggiore di", "LESS": "è minore di",
    "CONTAINS": "contiene", "INSIDE": "è contenuto in",
    "BEFORE": "avviene prima di", "AFTER": "avviene dopo di",
    "OPPOSITE": "è l'opposto di",
    "DURING": "avviene durante", "CONTAINS_TIME": "comprende temporalmente"
}

# --- CORE LOGIC: IL MOTORE RELAZIONALE ---

class RelationalGraph:
    def __init__(self):
        # adj: Mappa rapida {source: {target: (relation, derived_from)}}
        self.adj = {} 
        # edges: Lista piatta per iterazione regole
        self.edges = [] 

    def add_edge(self, source, relation, target, derived_from=None):
        """Aggiunge un arco e gestisce l'idempotenza."""
        if source in self.adj and target in self.adj[source]:
            curr_rel, _ = self.adj[source][target]
            if curr_rel == relation:
                return False # Arco già esistente

        if source not in self.adj: self.adj[source] = {}
        
        # Salvataggio nel grafo
        self.adj[source][target] = (relation, derived_from)
        
        edge_obj = {
            "source": source, "relation": relation, "target": target, 
            "derived_from": derived_from
        }
        self.edges.append(edge_obj)

        # Mutual Entailment (Generazione automatica dell'inverso)
        # Nota: L'inverso non ha 'derived_from' diretto per evitare loop nel backtracking
        inv_rel = INVERSE_RELATIONS.get(relation)
        if inv_rel:
            if target not in self.adj: self.adj[target] = {}
            if source not in self.adj[target]:
                self.adj[target][source] = (inv_rel, None) 
        
        return True

    def get_transitive_rule(self, r1, r2):
        """
        Matrice di Inferenza Logica (Combinatorial Entailment).
        Input: A--r1-->B e B--r2-->C
        Output: Relazione A-->C
        """
        # 1. Identità
        if r1 == "EQUALS": return r2
        if r2 == "EQUALS": return r1
        
        # 2. Comparazione
        if r1 == "GREATER" and r2 == "GREATER": return "GREATER"
        if r1 == "LESS" and r2 == "LESS": return "LESS"
        
        # 3. Gerarchia
        if r1 == "CONTAINS" and r2 == "CONTAINS": return "CONTAINS"
        if r1 == "INSIDE" and r2 == "INSIDE": return "INSIDE"
        
        # 4. Temporale
        if r1 == "BEFORE" and r2 == "BEFORE": return "BEFORE"
        if r1 == "AFTER" and r2 == "AFTER": return "AFTER"
        if r1 == "DURING" and r2 == "BEFORE": return "BEFORE" 
        
        # 5. Opposizione (La logica del doppio negativo)
        if r1 == "OPPOSITE" and r2 == "OPPOSITE": return "EQUALS"
        
        # 6. Distinzione (Propagazione dell'uguaglianza negativa)
        if r1 == "DISTINCT" and r2 == "EQUALS": return "DISTINCT"
        if r1 == "EQUALS" and r2 == "DISTINCT": return "DISTINCT"
        
        return None

    def compute_closure(self):
        """Calcola la chiusura transitiva completa del grafo."""
        changed = True
        while changed:
            changed = False
            # Copia shallow per iterare in sicurezza
            snapshot = list(self.edges)
            for e1 in snapshot:
                for e2 in snapshot:
                    # Logica: Se A->B e B->C, allora deduci A->C
                    if e1["target"] == e2["source"] and e1["source"] != e2["target"]:
                        new_rel = self.get_transitive_rule(e1["relation"], e2["relation"])
                        
                        if new_rel:
                            # Tenta di aggiungere l'arco derivato
                            # derived_from salva i "genitori" per il feedback
                            if self.add_edge(e1["source"], new_rel, e2["target"], derived_from=(e1, e2)):
                                changed = True

    def trace_logic_recursive(self, edge_obj, depth=0):
        """
        Deep Backtracking: Ricostruisce la catena logica fino alle premesse.
        """
        if not edge_obj: return []
        
        # Caso Base: È un Assioma (Premessa iniziale del Frame)
        if not edge_obj.get("derived_from"):
            return [f"• Dato che inizialmente abbiamo stabilito che: {edge_obj['source']} {RELATION_TEXT.get(edge_obj['relation'], 'rel')} {edge_obj['target']}."]

        parent1, parent2 = edge_obj["derived_from"]
        
        trace = []
        # Risalita ricorsiva (Depth-First Search nell'albero delle cause)
        trace.extend(self.trace_logic_recursive(parent1, depth+1))
        trace.extend(self.trace_logic_recursive(parent2, depth+1))
        
        # Deduzione locale
        deduction = (f"→ Poiché {parent1['source']} è legato a {parent1['target']} e "
                     f"{parent2['source']} a {parent2['target']}, "
                     f"ne consegue logicamente che: {edge_obj['source']} {RELATION_TEXT.get(edge_obj['relation'], 'rel')} {edge_obj['target']}.")
        
        trace.append(deduction)
        return trace

    def generate_feedback_deep(self, source, target, user_rel):
        """Genera il testo finale per l'utente."""
        if target not in self.adj[source]:
            return "Errore topologico: Non c'è alcuna evidenza logica che colleghi questi due elementi nel contesto attuale."
        
        real_rel, _ = self.adj[source][target]
        
        # Recupera l'oggetto edge specifico per avviare il trace
        edge_obj = next((e for e in self.edges if e["source"] == source and e["target"] == target and e["relation"] == real_rel), None)
        
        steps = self.trace_logic_recursive(edge_obj)
        # Rimuovi duplicati preservando l'ordine
        unique_steps = list(dict.fromkeys(steps))
        path_str = "\n".join(unique_steps)
        
        return (f"Errore. Hai ipotizzato: '{RELATION_TEXT.get(user_rel, user_rel)}'.\n\n"
                f"Ragionamento corretto:\n{path_str}\n\n"
                f"Conclusione: La relazione reale è '{RELATION_TEXT[real_rel]}'.")

    def to_dict(self):
        """Serializzazione JSON-ready."""
        return {"edges": self.edges}

    @staticmethod
    def from_dict(data):
        """Deserializzazione."""
        g = RelationalGraph()
        for e in data["edges"]:
            g.add_edge(e["source"], e["relation"], e["target"], e["derived_from"])
        return g

# --- DATASET: 50 FRAMES LOGICI ---

COMMON_GHOSTS = [
    "Zorp", "Glim", "Dax", "Plit", "Vra", "Klon", "Mir", "Flux", "Bel", 
    "Skree", "Bloop", "Juv", "Kax", "Lir", "Gloo", "Trak", "Vop", "Nif", "Mux", 
    "Rux", "Fin", "Teeb", "Ploy", "Gra", "Zaz", "Vreen", "Krot", "Frisp", "Dwap", 
    "Lorp", "Mish", "Gax", "Trok", "Sul", "Bep", "Zeek", "Plom", "Jik", "Sklo", 
    "Raf", "Tiz", "Vem", "Quip", "Zog", "Vax", "Mer", "Dul", "Glip", "Flurn", 
    "Scrit", "Yor", "Kin", "Waz", "Grak", "Zim", "Lul", "Praz", "Voop", "Yin",
    "Jaj", "Mim", "Zif", "Nod", "Pem", "Goy", "Snot", "Vel", "Krik", "Glo", "Bla"
]

FRAMES_DB = {
    # --- COORDINAZIONE (REL-COO) ---
    "REL-COO-001": {"vars": ["A", "B"], "premises": [("A", "EQUALS", "B")]},
    "REL-COO-002": {"vars": ["A", "B", "C"], "premises": [("A", "EQUALS", "B"), ("B", "EQUALS", "C")]},
    "REL-COO-003": {"vars": ["A", "B", "C", "D"], "premises": [("A", "EQUALS", "B"), ("C", "EQUALS", "D"), ("B", "EQUALS", "C")]},
    "REL-COO-004": {"vars": ["A", "B", "C", "D"], "premises": [("A", "EQUALS", "B"), ("C", "EQUALS", "D"), ("A", "DISTINCT", "C")]},
    "REL-COO-005": {"vars": ["A", "B", "C", "D", "E"], "premises": [("A", "EQUALS", "B"), ("B", "EQUALS", "C"), ("C", "EQUALS", "D"), ("D", "EQUALS", "E")]},
    "REL-COO-006": {"vars": ["A", "B"], "premises": [("A", "EQUALS", "B")]},

    # --- OPPOSIZIONE (REL-OPP) ---
    "REL-OPP-001": {"vars": ["A", "B"], "premises": [("A", "OPPOSITE", "B")]},
    "REL-OPP-002": {"vars": ["A", "B", "C"], "premises": [("A", "OPPOSITE", "B"), ("B", "OPPOSITE", "C")]},
    "REL-OPP-003": {"vars": ["A", "B", "C"], "premises": [("A", "EQUALS", "B"), ("B", "OPPOSITE", "C")]},
    "REL-OPP-004": {"vars": ["A", "B", "C", "D"], "premises": [("A", "OPPOSITE", "B"), ("C", "OPPOSITE", "D"), ("A", "EQUALS", "C")]},
    "REL-OPP-005": {"vars": ["A", "B", "C", "D"], "premises": [("A", "OPPOSITE", "B"), ("C", "OPPOSITE", "D"), ("B", "OPPOSITE", "C")]},
    "REL-OPP-006": {"vars": ["A", "B", "C"], "premises": [("A", "OPPOSITE", "B"), ("C", "OPPOSITE", "B")]},

    # --- DISTINZIONE (REL-DIS) ---
    "REL-DIS-001": {"vars": ["A", "B"], "premises": [("A", "DISTINCT", "B")]},
    "REL-DIS-002": {"vars": ["A", "B", "C"], "premises": [("A", "DISTINCT", "B"), ("B", "DISTINCT", "C")]},
    "REL-DIS-003": {"vars": ["A", "B", "C"], "premises": [("A", "EQUALS", "B"), ("B", "DISTINCT", "C")]},
    "REL-DIS-004": {"vars": ["A", "B", "C", "D"], "premises": [("A", "DISTINCT", "B"), ("B", "EQUALS", "C"), ("D", "DISTINCT", "C")]},
    "REL-DIS-005": {"vars": ["A", "B", "C", "D"], "premises": [("A", "DISTINCT", "B"), ("C", "DISTINCT", "D"), ("A", "EQUALS", "C")]},

    # --- CONFRONTO (REL-COM) ---
    "REL-COM-001": {"vars": ["A", "B"], "premises": [("A", "GREATER", "B")]},
    "REL-COM-002": {"vars": ["A", "B"], "premises": [("A", "LESS", "B")]},
    "REL-COM-003": {"vars": ["A", "B", "C"], "premises": [("A", "GREATER", "B"), ("B", "GREATER", "C")]},
    "REL-COM-004": {"vars": ["A", "B", "C"], "premises": [("A", "GREATER", "B"), ("C", "LESS", "B")]},
    "REL-COM-005": {"vars": ["A", "B", "C"], "premises": [("A", "EQUALS", "B"), ("B", "GREATER", "C")]},
    "REL-COM-006": {"vars": ["A", "B", "C", "D"], "premises": [("A", "GREATER", "B"), ("B", "GREATER", "C"), ("C", "GREATER", "D")]},
    "REL-COM-007": {"vars": ["A", "B", "C", "D"], "premises": [("A", "GREATER", "B"), ("C", "LESS", "D"), ("B", "EQUALS", "C")]},
    "REL-COM-008": {"vars": ["A", "B"], "premises": [("A", "GREATER", "B")]},
    "REL-COM-009": {"vars": ["A", "B", "C"], "premises": [("A", "GREATER", "B"), ("A", "GREATER", "C")]},
    "REL-COM-010": {"vars": ["A", "B", "C"], "premises": [("A", "GREATER", "B"), ("B", "GREATER", "C")]},
    "REL-COM-011": {"vars": ["A", "B", "C", "D"], "premises": [("A", "GREATER", "B"), ("B", "GREATER", "C"), ("D", "EQUALS", "A")]},

    # --- GERARCHIA (REL-HIE) ---
    "REL-HIE-001": {"vars": ["A", "B"], "premises": [("B", "INSIDE", "A")]},
    "REL-HIE-002": {"vars": ["A", "B", "C"], "premises": [("C", "INSIDE", "B"), ("B", "INSIDE", "A")]},
    "REL-HIE-003": {"vars": ["A", "B"], "premises": [("B", "INSIDE", "A")]},
    "REL-HIE-004": {"vars": ["A", "B", "C"], "premises": [("A", "CONTAINS", "B"), ("C", "DISTINCT", "A")]},
    "REL-HIE-005": {"vars": ["A", "B", "C"], "premises": [("A", "CONTAINS", "B"), ("A", "CONTAINS", "C"), ("B", "DISTINCT", "C")]},
    "REL-HIE-006": {"vars": ["A", "B", "C"], "premises": [("A", "CONTAINS", "B"), ("A", "CONTAINS", "C")]},
    "REL-HIE-007": {"vars": ["A", "B"], "premises": [("A", "INSIDE", "B")]},
    "REL-HIE-008": {"vars": ["A", "B", "C", "D"], "premises": [("A", "INSIDE", "B"), ("C", "INSIDE", "A"), ("D", "DISTINCT", "B")]},

    # --- TEMPORALE (REL-TEM) ---
    "REL-TEM-001": {"vars": ["A", "B"], "premises": [("A", "BEFORE", "B")]},
    "REL-TEM-002": {"vars": ["A", "B", "C"], "premises": [("A", "BEFORE", "B"), ("B", "BEFORE", "C")]},
    "REL-TEM-003": {"vars": ["A", "B"], "premises": [("A", "EQUALS", "B")]},
    "REL-TEM-004": {"vars": ["A", "B", "C"], "premises": [("A", "BEFORE", "B"), ("C", "AFTER", "B")]},
    "REL-TEM-005": {"vars": ["A", "B", "C"], "premises": [("A", "DURING", "B"), ("B", "BEFORE", "C")]},
    "REL-TEM-006": {"vars": ["A", "B", "C"], "premises": [("A", "CONTAINS_TIME", "B"), ("B", "BEFORE", "C")]},
    "REL-TEM-007": {"vars": ["A", "B", "C", "D"], "premises": [("A", "AFTER", "B"), ("A", "BEFORE", "C"), ("D", "AFTER", "C")]},

    # --- DEITTICA (REL-DEI) ---
    "REL-DEI-001": {"vars": ["A", "B"], "premises": [("A", "DISTINCT", "B")]},
    "REL-DEI-002": {"vars": ["A", "B"], "premises": [("A", "OPPOSITE", "B")]},
    "REL-DEI-003": {"vars": ["X", "Y"], "premises": [("X", "AFTER", "Y")]},
    "REL-DEI-004": {"vars": ["A", "B"], "premises": [("A", "OPPOSITE", "B")]},
    "REL-DEI-005": {"vars": ["A", "B"], "premises": [("A", "BEFORE", "B")]},
    "REL-DEI-006": {"vars": ["A", "B"], "premises": [("A", "EQUALS", "B")]},
    "REL-DEI-007": {"vars": ["A", "B"], "premises": [("A", "OPPOSITE", "B")]}
}

# --- SESSION MANAGER ---

class SessionManager:
    @staticmethod
    def save(session_id, graph):
        """Salva il grafo serializzato su Redis con TTL."""
        if USE_REDIS:
            data = json.dumps(graph.to_dict())
            redis_client.setex(f"session:{session_id}", SESSION_TTL, data)
        else:
            memory_store[session_id] = json.dumps(graph.to_dict())

    @staticmethod
    def load(session_id):
        """Carica il grafo da Redis."""
        data = None
        if USE_REDIS:
            data = redis_client.get(f"session:{session_id}")
        else:
            data = memory_store.get(session_id)
        
        if not data: return None
        return RelationalGraph.from_dict(json.loads(data))

# --- LOGGING ASINCRONO ---

def log_interaction_async(session_id: str, is_valid: bool, time_ms: int, spatial: Optional[SpatialData]):
    """
    Task asincrono per loggare dati di performance e spaziali
    senza bloccare la risposta HTTP del motore logico.
    Qui andrebbe la chiamata a MongoDB/PostgreSQL.
    """
    # print(f"📝 [LOG] Sess: {session_id} | OK: {is_valid} | {time_ms}ms | {spatial}")
    pass

# --- API ENDPOINTS ---

@app.get("/session/new/{frame_id}")
async def init_session(frame_id: str):
    """
    Inizializza una nuova sessione di gioco.
    1. Pesca il frame.
    2. Astrae le entità (A,B -> Zorp,Glim).
    3. Calcola tutte le inferenze possibili (Chiusura).
    4. Salva su Redis.
    """
    if frame_id not in FRAMES_DB:
        raise HTTPException(status_code=404, detail=f"Frame ID {frame_id} non trovato.")
    
    frame = FRAMES_DB[frame_id]
    
    # Randomizzazione Variabili Ghost
    if len(COMMON_GHOSTS) < len(frame["vars"]):
        raise HTTPException(500, "Errore interno: Pool variabili insufficiente.")
        
    ghosts = random.sample(COMMON_GHOSTS, len(frame["vars"]))
    var_map = dict(zip(frame["vars"], ghosts))
    
    graph = RelationalGraph()
    frontend_premises = []
    
    # Costruzione Grafo iniziale
    for p in frame["premises"]:
        p_src, p_rel, p_tgt = p
        g_src = var_map[p_src]
        g_tgt = var_map[p_tgt]
        
        graph.add_edge(g_src, p_rel, g_tgt)
        frontend_premises.append({"source": g_src, "relation": p_rel, "target": g_tgt})
    
    # Calcolo Chiusura Immediata (<5ms)
    graph.compute_closure()
    
    # Generazione Sessione
    session_id = str(uuid.uuid4())[:8]
    SessionManager.save(session_id, graph)
    
    return {
        "session_id": session_id,
        "nodes": ghosts,
        "premises": frontend_premises,
        "frame_info": {"id": frame_id, "type": frame_id.split("-")[1]}
    }

@app.post("/validate-edge")
async def validate(req: ValidationRequest, background_tasks: BackgroundTasks):
    """
    Valida un arco tracciato dall'utente.
    Se errato, restituisce il Logic Path (Deep Backtracking).
    """
    graph = SessionManager.load(req.session_id)
    if not graph:
        raise HTTPException(404, "Sessione scaduta o inesistente.")
    
    u_s, u_r, u_t = req.user_edge.source, req.user_edge.relation, req.user_edge.target
    
    is_valid = False
    feedback = None
    expected = None
    
    # Check Topologico
    if u_s in graph.adj and u_t in graph.adj[u_s]:
        real_rel, _ = graph.adj[u_s][u_t]
        if real_rel == u_r:
            is_valid = True
        else:
            is_valid = False
            expected = real_rel
            # Generazione Feedback Ricorsivo
            feedback = graph.generate_feedback_deep(u_s, u_t, u_r)
    else:
        is_valid = False
        feedback = "Nessuna relazione logica diretta o derivata rilevata tra questi due nodi nel contesto attuale."

    # Logging asincrono
    background_tasks.add_task(log_interaction_async, req.session_id, is_valid, req.thinking_time_ms, req.spatial_data)
    
    if is_valid:
        return {"status": "success", "is_valid": True}
    else:
        return {
            "status": "error", 
            "is_valid": False, 
            "logic_path": feedback, 
            "expected_relation": expected
        }

@app.post("/session/reset")
async def reset_session(frame_id: str):
    """Utilità per il frontend dev: riavvia velocemente lo stesso livello."""
    return await init_session(frame_id)
