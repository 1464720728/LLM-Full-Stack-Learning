"""Explicit opt-in embeddings API adapter and exact cosine retrieval.

Default CLI is an offline protocol fixture, not a semantic quality demonstration.
--live requires EMBEDDING_API_KEY, EMBEDDING_BASE_URL and EMBEDDING_MODEL.
"""
import argparse
from dataclasses import asdict,dataclass
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
from urllib.error import HTTPError,URLError
from urllib.parse import urlsplit
from urllib.request import Request,build_opener,HTTPRedirectHandler

ADAPTER_VERSION='course-embeddings-v1'
MAX_RESPONSE_BYTES=2_000_000


class EmbeddingError(ValueError):pass


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        return None


def endpoint(base_url):
    parts=urlsplit(base_url)
    if parts.username or parts.password or parts.query or parts.fragment:
        raise EmbeddingError('base URL must not include credentials, query or fragment')
    local=parts.hostname in {'localhost','127.0.0.1','::1'}
    if not parts.hostname or not (parts.scheme=='https' or (parts.scheme=='http' and local)):
        raise EmbeddingError('HTTPS required except explicit localhost HTTP')
    return base_url.rstrip('/')+'/embeddings'


def http_transport(url,headers,payload,timeout):
    request=Request(url,data=json.dumps(payload,ensure_ascii=False).encode('utf-8'),headers=headers,method='POST')
    try:
        # Refuse redirects: never forward bearer credentials to a new endpoint.
        with build_opener(_NoRedirect()).open(request,timeout=timeout) as response:
            raw=response.read(MAX_RESPONSE_BYTES+1)
    except HTTPError as exc:
        raise EmbeddingError(f'embeddings HTTP {exc.code}; inspect provider settings without logging credentials') from None
    except (URLError,TimeoutError,OSError):
        raise EmbeddingError('embeddings network/timeout failure; no automatic retry was made') from None
    if len(raw)>MAX_RESPONSE_BYTES:
        raise EmbeddingError('response exceeds local size limit')
    try:
        return json.loads(raw)
    except (UnicodeDecodeError,json.JSONDecodeError):
        raise EmbeddingError('response is not valid UTF-8 JSON') from None


def validate_response(response,count,expected_dimension=None):
    if not isinstance(response,dict) or not isinstance(response.get('model'),str) or not response['model']:
        raise EmbeddingError('missing response model identity')
    data=response.get('data')
    if not isinstance(data,list) or len(data)!=count:
        raise EmbeddingError('response item count differs from request')
    rows=[None]*count
    dimension=expected_dimension
    for item in data:
        if not isinstance(item,dict):raise EmbeddingError('embedding item must be an object')
        index=item.get('index');vector=item.get('embedding')
        if type(index) is not int or not 0<=index<count or rows[index] is not None:
            raise EmbeddingError('invalid or duplicated response index')
        if not isinstance(vector,list) or not vector or len(vector)>65536:
            raise EmbeddingError('embedding must be a bounded nonempty float array')
        if any(type(x) not in (int,float) or not math.isfinite(x) for x in vector):
            raise EmbeddingError('embedding includes nonnumeric or nonfinite values')
        if dimension is None:dimension=len(vector)
        if len(vector)!=dimension:raise EmbeddingError('inconsistent embedding dimension')
        rows[index]=[float(x) for x in vector]
    usage=response.get('usage',{})
    if not isinstance(usage,dict):raise EmbeddingError('usage must be an object')
    tokens=usage.get('total_tokens')
    if tokens is not None and (type(tokens) is not int or tokens<0):raise EmbeddingError('invalid token usage')
    return rows,response['model'],tokens


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors:list[list[float]]
    metadata:dict


class EmbeddingClient:
    def __init__(self,base_url,model,api_key='',*,allow_network=False,transport=None,
                 batch_size=8,max_requests=4,max_total_chars=12000,timeout=20,dimensions=None,
                 model_revision='provider-alias-unpinned'):
        self.url=endpoint(base_url)
        if not isinstance(api_key,str) or '\r' in api_key or '\n' in api_key:
            raise EmbeddingError('invalid API key format; credentials were not logged')
        if not isinstance(model,str) or not model.strip():raise EmbeddingError('model required')
        if any(type(x) is not int or x<=0 for x in (batch_size,max_requests,max_total_chars)):
            raise EmbeddingError('positive integer limits required')
        if not math.isfinite(timeout) or not 0<timeout<=120:raise EmbeddingError('timeout must be in (0,120] seconds')
        if dimensions is not None and (type(dimensions) is not int or not 1<=dimensions<=65536):
            raise EmbeddingError('invalid requested dimension')
        if transport is None and not allow_network:
            raise EmbeddingError('real network requires explicit allow_network=True')
        if transport is None and not api_key:
            raise EmbeddingError('API key environment value is missing')
        self.model,self._api_key=model,api_key
        self.transport=transport or http_transport
        self.mode='live' if transport is None else 'injected-offline-transport'
        self.batch_size,self.max_requests,self.max_total_chars=batch_size,max_requests,max_total_chars
        self.timeout,self.dimensions,self.model_revision=timeout,dimensions,model_revision

    def embed(self,texts):
        if not isinstance(texts,list) or not texts or any(not isinstance(t,str) or not t.strip() for t in texts):
            raise EmbeddingError('nonempty list of nonempty strings required')
        if sum(map(len,texts))>self.max_total_chars:raise EmbeddingError('local character budget exceeded')
        requests=(len(texts)+self.batch_size-1)//self.batch_size
        if requests>self.max_requests:raise EmbeddingError('request count budget exceeded before sending')
        vectors=[];response_model=None;tokens=0;usage_complete=True;dimension=self.dimensions
        for start in range(0,len(texts),self.batch_size):
            subset=texts[start:start+self.batch_size]
            payload={'input':subset,'model':self.model,'encoding_format':'float'}
            if self.dimensions is not None:payload['dimensions']=self.dimensions
            headers={'Content-Type':'application/json','Authorization':'Bearer '+self._api_key}
            response=self.transport(self.url,headers,payload,self.timeout)
            rows,returned_model,used=validate_response(response,len(subset),dimension)
            if response_model is not None and response_model!=returned_model:
                raise EmbeddingError('model identity changed between batches')
            response_model=returned_model;dimension=len(rows[0]);vectors.extend(rows)
            if used is None:usage_complete=False
            else:tokens+=used
        return EmbeddingBatch(vectors,{
            'adapter_version':ADAPTER_VERSION,'mode':self.mode,'base_endpoint':self.url,
            'requested_model':self.model,'response_model':response_model,'model_revision_note':self.model_revision,
            'dimension':dimension,'encoding_format':'float','metric':'cosine','request_count':requests,
            'reported_total_tokens':tokens if usage_complete else None,
            'reported_known_tokens':tokens,'usage_complete':usage_complete,
            'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'text_count':len(texts),'local_character_count':sum(map(len,texts)),
            'token_limit_note':'Character budget is not provider token accounting; tokenizer limits still apply.'})


def cosine_topk(vectors,query,k=3):
    if type(k) is not int or k<0:raise EmbeddingError('nonnegative integer k required')
    if not query or any(type(x) not in (int,float) or not math.isfinite(x) for x in query):
        raise EmbeddingError('query must be a finite nonempty numeric vector')
    if any(len(v)!=len(query) for v in vectors):raise EmbeddingError('dimension mismatch')
    if any(type(x) not in (int,float) or not math.isfinite(x) for v in vectors for x in v):
        raise EmbeddingError('document vectors must be finite numbers')
    def unit(vector):
        # Scaling first prevents finite 1e308 values overflowing on squaring,
        # and finite 1e-308 values underflowing into an apparent zero vector.
        scale=max(abs(x) for x in vector)
        if not scale:return None
        scaled=[x/scale for x in vector]
        norm=math.sqrt(math.fsum(x*x for x in scaled))
        return [x/norm for x in scaled]
    q=unit(query)
    if q is None:return []
    scored=[]
    for i,v in enumerate(vectors):
        normalized=unit(v)
        if normalized is not None:
            score=max(-1.0,min(1.0,math.fsum(a*b for a,b in zip(normalized,q))))
            scored.append((i,score))
    return sorted(scored,key=lambda item:(-item[1],item[0]))[:k]


def dense_search(store,context,query,client,k=3):
    """Embed only current authorized chunks and query, then exact cosine ranking.

    This transparent small-corpus path re-embeds on each call. A production cache
    must bind chunk revision, tenant, embedding identity and access policy.
    """
    from .retrieval import Hit
    if not isinstance(query,str) or not query.strip():raise EmbeddingError('nonempty query required')
    if type(k) is not int or k<0:raise EmbeddingError('nonnegative integer k required')
    chunks=store.visible(context)
    if not chunks:return [],{'mode':'no_visible_documents','request_count':0}
    batch=client.embed([c.text for c in chunks]+[query])
    ranking=cosine_topk(batch.vectors[:-1],batch.vectors[-1],k)
    # Recheck current authorization/version before returning stored snapshots.
    current={c.chunk_id for c in store.visible(context)}
    hits=[Hit(chunks[i],score) for i,score in ranking if chunks[i].chunk_id in current]
    return hits,batch.metadata


DEMO_DOCUMENTS=['A feline is a cat.','A canine is a dog.','KV cache stores attention keys and values.']
DEMO_QUERY='Which document describes a kitten or cat?'


def mock_transport(url,headers,payload,timeout):
    # Hand-assigned fixtures test response plumbing. They are not learned embeddings.
    table={DEMO_DOCUMENTS[0]:[1.0,0.0,0.0],DEMO_DOCUMENTS[1]:[0.0,1.0,0.0],
           DEMO_DOCUMENTS[2]:[0.0,0.0,1.0],DEMO_QUERY:[0.9,0.1,0.0]}
    return {'model':'offline-vector-fixture-v1','usage':{'total_tokens':0},
            'data':[{'index':i,'embedding':table[t]} for i,t in reversed(list(enumerate(payload['input'])))]}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true',help='send four fictional inputs to configured endpoint')
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--max-requests',type=int,default=4)
    parser.add_argument('--dimensions',type=int)
    parser.add_argument('--output',help='optional local JSON result path; contains vectors, not API keys')
    args=parser.parse_args(argv)
    if args.output:
        destination=Path(args.output)
        if destination.exists() or not destination.parent.is_dir():
            parser.error('output must be a new file in an existing directory; no API call was made')
    if args.live:
        required={name:os.environ.get(name,'') for name in ('EMBEDDING_API_KEY','EMBEDDING_BASE_URL','EMBEDDING_MODEL')}
        if not all(required.values()):parser.error('set EMBEDDING_API_KEY, EMBEDDING_BASE_URL and EMBEDDING_MODEL')
        client=EmbeddingClient(required['EMBEDDING_BASE_URL'],required['EMBEDDING_MODEL'],required['EMBEDDING_API_KEY'],
            allow_network=True,batch_size=args.batch_size,max_requests=args.max_requests,dimensions=args.dimensions,
            model_revision=os.environ.get('EMBEDDING_MODEL_REVISION','provider-alias-unpinned'))
    else:
        if args.dimensions is not None:parser.error('--dimensions is only used for an explicitly live request')
        client=EmbeddingClient('https://offline.invalid/v1','offline-fixture',transport=mock_transport,
                               batch_size=args.batch_size,max_requests=args.max_requests)
    try:
        batch=client.embed(DEMO_DOCUMENTS+[DEMO_QUERY])
        result={'metadata':batch.metadata,'documents':DEMO_DOCUMENTS,'query':DEMO_QUERY,
                'document_vectors':batch.vectors[:-1],'query_vector':batch.vectors[-1],
                'ranking':[{'document_id':i,'text':DEMO_DOCUMENTS[i],'cosine':score}
                           for i,score in cosine_topk(batch.vectors[:-1],batch.vectors[-1])],
                'notice':'Live mode uses provider embeddings; default uses synthetic protocol fixtures only.'}
    except EmbeddingError as exc:parser.exit(2,str(exc)+'\n')
    encoded=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output:
        path=Path(args.output)
        # Exclusive create avoids silently replacing an earlier experiment.
        with path.open('x',encoding='utf-8') as f:f.write(encoded+'\n')
    print(encoded)


if __name__=='__main__':main()
