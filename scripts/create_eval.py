"""Curated development queries. Does not use model predictions to choose labels."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
items=[
('e01','exact','What does response_model_exclude_unset do?', [('tutorial/response-model.md','return only the values explicitly set')]),
('e02','exact','Header convert_underscores False', [('tutorial/header-params.md','disable automatic conversion of underscores to hyphens')]),
('e03','exact','CORSMiddleware allow_credentials default', [('tutorial/cors.md','cookies should be supported for cross-origin requests')]),
('e04','exact','CORS max_age default 600', [('tutorial/cors.md','Defaults to `600`')]),
('e05','exact','OAuth2PasswordBearer tokenUrl relative URL', [('tutorial/security/first-steps.md','refers to a relative URL')]),
('e06','exact','WebSocketDisconnect receive_text exception', [('advanced/websockets.md','When a WebSocket connection is closed')]),
('e07','exact','UploadFile SpooledTemporaryFile object', [('tutorial/request-files.md','It exposes an actual Python')]),
('e08','exact','BackgroundTasks add_task arguments', [('tutorial/background-tasks.md','`.add_task()` receives as arguments')]),
('e09','exact','Path ge=1 item_id validation', [('tutorial/path-params-numeric-validations.md','Here, with `ge=1`')]),
('e10','exact','status_code HTTPStatus IntEnum', [('tutorial/response-status-code.md','can alternatively also receive an `IntEnum`')]),
('c01','conceptual','How can I send an email after replying so the visitor does not have to wait?', [('tutorial/background-tasks.md','after* returning a response')]),
('c02','conceptual','How can a browser website call a server running on a different port?', [('tutorial/cors.md','An origin is the combination of protocol')]),
('c03','conceptual','How do I keep large uploaded attachments from exhausting RAM?', [('tutorial/request-files.md','It uses a "spooled" file')]),
('c04','conceptual','How do I close my database session automatically when request handling finishes?', [('tutorial/dependencies/dependencies-with-yield.md','create a database session and close it after finishing')]),
('c05','conceptual','How can I avoid leaking private fields in the data returned to a client?', [('tutorial/response-model.md','limit and filter** the output data')]),
('c06','conceptual','How should a chat server handle a participant disconnecting?', [('advanced/websockets.md','When a WebSocket connection is closed')]),
('c07','conceptual','Where should I load a shared machine learning model once and release it when the service stops?', [('advanced/events.md','database connection pool, or loading a shared machine learning model')]),
('c08','conceptual','How do I reject a request and return an explanation when something is missing?', [('tutorial/handling-errors.md','you don\'t `return` it, you `raise` it')]),
('c09','conceptual','How can I reuse common request validation across different routes?', [('tutorial/dependencies/index.md','shared logic')]),
('c10','conceptual','How do I stop overly long text from being accepted as an input?', [('tutorial/query-params-str-validations.md',"its length doesn't exceed 50 characters")]),
('c11','conceptual','How can my application understand an object embedded inside another object?', [('tutorial/body-nested-models.md','nested models')]),
('c12','conceptual','How can the browser obtain proof of identity and attach it to later requests?', [('tutorial/security/first-steps.md','The frontend stores that token temporarily somewhere')]),
('c13','conceptual','How can I avoid buffering a large attachment and notify the user after replying?', [('tutorial/request-files.md','It uses a "spooled" file'),('tutorial/background-tasks.md','after* returning a response')]),
('c14','conceptual','How can I clean up resources both for an individual request and when the application exits?', [('tutorial/dependencies/dependencies-with-yield.md','create a database session and close it after finishing'),('advanced/events.md','database connection pool, or loading a shared machine learning model')]),
]
output=[]
for qid,kind,query,targets in items:
 labels=[]
 for doc,quote in targets:
  text=(ROOT/'corpus'/doc).read_text()
  if quote not in text:
   print('MISSING',qid,doc,repr(quote));continue
  labels.append({'doc':doc,'quote':quote,'occurrence':0})
 if len(labels)!=len(targets):raise SystemExit('Fix missing evidence before writing labels')
 output.append({'query_id':qid,'query':query,'kind':kind,'relevant':labels,
                'rationale':'Exact identifier lookup.' if kind=='exact' else 'Natural-language intent; evidence selected from the source documentation before running retrieval.'})
(ROOT/'data/eval_queries.json').write_text(json.dumps(output,indent=2)+'\n')
print('Curated',len(output),'queries with',sum(len(x['relevant']) for x in output),'evidence spans')
