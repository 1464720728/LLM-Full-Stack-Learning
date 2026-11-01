import unittest
from unittest.mock import patch
from labs.knowledge.embedding_api import (EmbeddingClient,EmbeddingError,endpoint,validate_response,
    cosine_topk,mock_transport,DEMO_DOCUMENTS,DEMO_QUERY)


def response(vectors,model='test-model'):
    return {'model':model,'data':[{'index':i,'embedding':v} for i,v in enumerate(vectors)],'usage':{'total_tokens':2}}


class EmbeddingAdapterTests(unittest.TestCase):
    def test_network_is_opt_in(self):
        with self.assertRaises(EmbeddingError):EmbeddingClient('https://example.invalid/v1','m','test-only-key')
    def test_secret_with_newline_rejected_without_echo(self):
        with self.assertRaises(EmbeddingError) as caught:
            EmbeddingClient('https://example.invalid/v1','m','fake-secret\n',allow_network=True)
        self.assertNotIn('fake-secret',str(caught.exception))
    def test_no_insecure_remote(self):
        with self.assertRaises(EmbeddingError):endpoint('http://example.invalid/v1')
        self.assertEqual(endpoint('http://127.0.0.1:1234/v1'),'http://127.0.0.1:1234/v1/embeddings')
    def test_credentials_in_url_rejected(self):
        with self.assertRaises(EmbeddingError):endpoint('https://name:password@example.invalid/v1')
    def test_response_reordered_by_index(self):
        payload=response([[1,0],[0,1]]);payload['data'].reverse()
        self.assertEqual(validate_response(payload,2)[0],[[1.0,0.0],[0.0,1.0]])
    def test_duplicate_index_rejected(self):
        payload=response([[1,0],[0,1]]);payload['data'][1]['index']=0
        with self.assertRaises(EmbeddingError):validate_response(payload,2)
    def test_nan_rejected(self):
        with self.assertRaises(EmbeddingError):validate_response(response([[float('nan')]]),1)
    def test_missing_usage_is_unknown(self):
        for usage in [None,{}, {'prompt_tokens':2}]:
            payload=response([[1,0]])
            if usage is None:payload.pop('usage')
            else:payload['usage']=usage
            self.assertIsNone(validate_response(payload,1)[2])
    def test_partial_batch_usage_does_not_claim_known_total(self):
        calls=[]
        def transport(*args):
            payload=response([[1,0]])
            if calls:payload.pop('usage')
            calls.append(1)
            return payload
        client=EmbeddingClient('https://example.invalid/v1','m',transport=transport,batch_size=1)
        meta=client.embed(['a','b']).metadata
        self.assertIsNone(meta['reported_total_tokens'])
        self.assertFalse(meta['usage_complete'])
        self.assertEqual(meta['reported_known_tokens'],2)
    def test_zero_reported_usage_is_distinct_from_unknown(self):
        client=EmbeddingClient('https://offline.invalid/v1','fixture',transport=mock_transport)
        meta=client.embed([DEMO_QUERY]).metadata
        self.assertEqual(meta['reported_total_tokens'],0)
        self.assertTrue(meta['usage_complete'])
    def test_boolean_rejected(self):
        with self.assertRaises(EmbeddingError):validate_response(response([[True]]),1)
    def test_dimension_change_rejected(self):
        with self.assertRaises(EmbeddingError):validate_response(response([[1,2],[1,2,3]]),2)
    def test_missing_item_rejected(self):
        with self.assertRaises(EmbeddingError):validate_response(response([[1,2]]),2)
    def test_budget_rejected_before_transport(self):
        calls=[]
        client=EmbeddingClient('https://example.invalid/v1','m',transport=lambda *a:calls.append(a),batch_size=1,max_requests=1)
        with self.assertRaises(EmbeddingError):client.embed(['one','two'])
        self.assertEqual(calls,[])
    def test_complete_offline_plumbing(self):
        client=EmbeddingClient('https://offline.invalid/v1','fixture',transport=mock_transport,batch_size=2)
        batch=client.embed(DEMO_DOCUMENTS+[DEMO_QUERY])
        self.assertEqual(batch.metadata['request_count'],2)
        self.assertEqual(batch.metadata['dimension'],3)
        self.assertEqual(cosine_topk(batch.vectors[:-1],batch.vectors[-1])[0][0],0)
        self.assertNotIn('api_key',batch.metadata)
    def test_cross_batch_model_change(self):
        count=0
        def transport(*args):
            nonlocal count
            count+=1
            return response([[1,0]],f'model-{count}')
        client=EmbeddingClient('https://example.invalid/v1','m',transport=transport,batch_size=1)
        with self.assertRaises(EmbeddingError):client.embed(['a','b'])
    def test_query_dimension_guard(self):
        with self.assertRaises(EmbeddingError):cosine_topk([[1,2]],[1])
    def test_zero_query_no_arbitrary_result(self):self.assertEqual(cosine_topk([[1,0]],[0,0]),[])
    def test_cosine_extreme_finite_scales(self):
        import math
        for scale in [1e308,1e-308,5e-324]:
            ranking=cosine_topk([[scale,scale],[-scale,-scale],[0,0]],[scale,scale])
            self.assertEqual([i for i,_ in ranking],[0,1])
            self.assertTrue(all(math.isfinite(score) for _,score in ranking))
            self.assertAlmostEqual(ranking[0][1],1)
            self.assertAlmostEqual(ranking[1][1],-1)
    def test_http_transport_injected_request(self):
        calls=[]
        def transport(url,headers,payload,timeout):
            calls.append((url,headers,payload,timeout))
            return response([[1,0]])
        client=EmbeddingClient('https://example.invalid/v1','requested-model','test-only-key',transport=transport,dimensions=2)
        client.embed(['hello'])
        self.assertEqual(calls[0][2],{'input':['hello'],'model':'requested-model','encoding_format':'float','dimensions':2})
        self.assertEqual(calls[0][1]['Authorization'],'Bearer test-only-key')
    def test_dense_search_does_not_send_hidden_document(self):
        from labs.knowledge.embedding_api import dense_search
        from labs.knowledge.store import CorpusStore,Context
        calls=[]
        def transport(url,headers,payload,timeout):
            calls.extend(payload['input'])
            return mock_transport(url,headers,payload,timeout)
        client=EmbeddingClient('https://offline.invalid/v1','fixture',transport=transport)
        with CorpusStore() as store:
            store.ingest('t','public','Public',DEMO_DOCUMENTS[0],('*',))
            store.ingest('t','hidden','Hidden','secret material',('group:admins',))
            hits,meta=dense_search(store,Context('t','learner'),DEMO_QUERY,client)
            self.assertEqual(hits[0].chunk.doc_id,'public')
            self.assertNotIn('secret material',calls)
    def test_faiss_optional_run(self):
        from labs.knowledge.faiss_lab import run
        result=run(n=50,dim=8,queries=5,k=3)
        if result['status']=='SKIP':self.skipTest(result['reason'])
        self.assertTrue(all(0<=row['ann_neighbor_recall_at_k']<=1 for row in result['measurements']))


if __name__=='__main__':unittest.main()
