"""Requires requirements.txt, but NO downloaded weights: a tiny random Llama fixture.

This suite is not evidence of the pretrained model's factual or semantic behavior.
"""
import unittest
import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig,LlamaForCausalLM,PreTrainedTokenizerFast
from engine import Engine,tensor_output


def fixture():
    words=['[UNK]','[EOS]','The','capital','of','Japan','is','Tokyo','France','Paris','Italy','Rome','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t']
    tok=Tokenizer(WordLevel({w:i for i,w in enumerate(words)},unk_token='[UNK]'));tok.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=tok,unk_token='[UNK]',eos_token='[EOS]')
    torch.manual_seed(11)
    config=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
                       num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=512,
                       tie_word_embeddings=True,bos_token_id=None,eos_token_id=1,attention_dropout=0.)
    config._attn_implementation='eager'
    return Engine(LlamaForCausalLM(config),tokenizer)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.e=fixture();cls.prompt='The capital of Japan is'
    def test_full_verification(self):
        d=self.e.verify(self.prompt);self.assertTrue(d['passed'],[c for c in d['checks'] if not c['passed']])
    def test_predict_precision(self):
        d=self.e.predict(self.prompt);self.assertEqual(len(d['logits']),32);self.assertTrue(any(x!=round(x,3) for x in d['logits']))
    def test_context_not_silently_truncated(self):
        with self.assertRaisesRegex(ValueError,'Nothing was truncated'):self.e.attention(' '.join(['a']*65))
    def test_empty_prompt(self):
        with self.assertRaises(ValueError):self.e.predict('')
    def test_inspector_all_positions_and_gqa_groups(self):
        for head in [1,2,3,4]:
            d=self.e.inspect(self.prompt,1,head,position=1)
            self.assertTrue(all(c['passed'] for c in d['checks']))
            self.assertEqual(d['shared_kv_head'],(head-1)//2+1)
            self.assertEqual(sum(r['weight'] for r in d['attention_row'][2:]),0)
            p=d['q_coordinate_0_projection'];self.assertAlmostEqual(sum(p['products'])+p['bias'],p['result'],5)
            u=d['unembedding'];self.assertAlmostEqual(sum(u['dot_terms']['values']),u['model_logit'],5)
    def test_raw_and_normalized_stages(self):
        d=self.e.lenses(self.prompt);self.assertEqual(len(d['stages']),4);self.assertIn('raw',d['stages'][-2]['label']);self.assertIn('already normalized',d['stages'][-1]['label']);self.assertTrue(d['checks'][0]['passed'])
    def test_seed_replay_and_distinct_streams(self):
        a=self.e.generate(self.prompt,4,.7,2,101,False,False);b=self.e.generate(self.prompt,4,.7,2,101,False,False)
        self.assertEqual([r['generated_ids'] for r in a['results']],[r['generated_ids'] for r in b['results']]);self.assertEqual([r['seed'] for r in a['results']],[101,102])
    def test_cached_generation_greedy(self):
        a=self.e.generate(self.prompt,3,0,1,3,False,False);b=self.e.generate(self.prompt,3,0,1,3,True,False)
        self.assertEqual(a['results'][0]['generated_ids'],b['results'][0]['generated_ids']);self.assertEqual([s['processed_positions'] for s in b['results'][0]['steps']],[5,1,1])
    def test_uniform_intervention_independent_oracle(self):
        e=self.e;ids=e.encode(self.prompt)
        with torch.no_grad():
            _,states,_,_=e.capture(ids);m=e.manual_block(states[0],0);v=m['v'].repeat_interleave(e.group_size,dim=1)
            uv=v.cumsum(2)/torch.arange(1,ids.shape[1]+1).view(1,1,-1,1)
            independent_write=e.model.model.layers[0].self_attn.o_proj(uv.transpose(1,2).reshape(1,ids.shape[1],e.width))
            for scope in ['last','all']:
                def replace(mod,args,out):
                    write=tensor_output(out).clone()
                    if scope=='all':write[:]=independent_write
                    else:write[:,-1]=independent_write[:,-1]
                    return (write,)+out[1:]
                handle=e.model.model.layers[0].self_attn.register_forward_hook(replace)
                try:oracle=e.model(ids,use_cache=False).logits[0,-1]
                finally:handle.remove()
                actual=e.modified_logits(ids,0,tuple(range(e.heads)),scope,True)
                torch.testing.assert_close(actual,oracle,atol=2e-4,rtol=2e-4)
    def test_intervention_metrics_and_restore(self):
        before=self.e.predict(self.prompt)['logits']
        for uniform in [False,True]:
            d=self.e.knockout(self.prompt,1,'all',7,9,uniform);self.assertEqual(len(d['rows']),5);self.assertTrue(d['checks'][0]['passed'])
            for r in d['rows']:self.assertGreaterEqual(r['tv'],0);self.assertLessEqual(r['tv'],1)
        np.testing.assert_array_equal(before,self.e.predict(self.prompt)['logits'])
    def test_hook_cleanup_on_forward_exception(self):
        e=self.e;layer=e.model.model.layers[-1]
        def fail(*args):raise RuntimeError('injected test failure')
        handle=layer.mlp.register_forward_hook(fail)
        try:
            with self.assertRaisesRegex(RuntimeError,'injected'):e.modified_logits(e.encode(self.prompt),0,(0,),uniform=True)
        finally:handle.remove()
        for block in e.model.model.layers:
            self.assertEqual(len(block.self_attn.o_proj._forward_pre_hooks),0);self.assertEqual(len(block.self_attn.v_proj._forward_hooks),0)
    def test_teacher_forcing_matches_model_loss(self):
        d=self.e.training_trace(self.prompt);ids=self.e.encode(self.prompt)
        with torch.no_grad():loss=float(self.e.model(ids,labels=ids,use_cache=False).loss)
        self.assertEqual(len(d['rows']),4);self.assertAlmostEqual(d['mean_loss_nats'],loss,5)
    def test_comparison_identity(self):
        d=self.e.compare(self.prompt,self.prompt,[7,9],1,1);self.assertEqual(d['metrics']['tv'],0);self.assertTrue(all(r['delta_pp']==0 for r in d['candidates']))
    def test_affine_probe_and_overlap_rejection(self):
        d=self.e.fit_probe(['a b','c d','e f','g h'],['i j','k l']);self.assertEqual(len(d['scores']),3);self.assertLess(d['scores'][-1]['probe_kl'],1e-6)
        with self.assertRaises(ValueError):self.e.fit_probe(['a b','c d','e f','g h'],['a b','k l'])
        with self.assertRaises(ValueError):self.e.fit_probe(['a b','c d','e f','g h'],['a  b','k l'])
    def test_local_jvp_and_final_identity(self):
        for layer in [1,2]:
            d=self.e.jacobian(self.prompt,layer,3,.01);self.assertTrue(all(c['passed'] for c in d['checks']),d['checks'])
            if layer==2:np.testing.assert_allclose(d['jvp'],np.eye(16)[3])
    def test_provenance_and_frozen_weights(self):
        self.assertTrue(self.e.provenance['tied_weights_actual']);self.assertIn('UNTRAINED',self.e.provenance['model_kind']);self.assertTrue(all(not p.requires_grad for p in self.e.model.parameters()))


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        import main
        cls.main=main;main._engine=fixture();cls.client=TestClient(main.app)
    def test_static_assets(self):
        for path in ['/','/app.js','/math.js','/style.css']:self.assertEqual(self.client.get(path).status_code,200,path)
    def test_api_endpoints(self):
        for path in ['/api/health','/api/tokenize?prompt=The','/api/decode?id=7','/api/toy-attention','/api/toy-training']:self.assertEqual(self.client.get(path).status_code,200,path)
        payloads={'predict':{'prompt':'a b'},'attention':{'prompt':'a b'},'inspect':{'prompt':'a b','layer':1,'head':1},'generate':{'prompt':'a b','n_tokens':2},'lenses':{'prompt':'a b'},'intervene':{'prompt':'a b','layer':1},'training-trace':{'prompt':'a b'},'compare':{'prompt_a':'a b','prompt_b':'c d','layer':1},'jacobian':{'prompt':'a b','layer':1},'verify':{'prompt':'a b'},'probe-fit':{'train_prompts':['a b','c d','e f','g h'],'test_prompts':['i j','k l']}}
        for path,payload in payloads.items():
            response=self.client.post('/api/'+path,json=payload);self.assertEqual(response.status_code,200,(path,response.text[:500]))
    def test_validation(self):
        for path,payload in [('predict',{'prompt':''}),('generate',{'prompt':'a','temperature':-1}),('inspect',{'prompt':'a','layer':0,'head':1}),('intervene',{'prompt':'a','layer':1,'scope':'invalid'})]:self.assertEqual(self.client.post('/api/'+path,json=payload).status_code,422)
    def test_untrusted_host(self):self.assertEqual(self.client.get('/api/health',headers={'host':'evil.example'}).status_code,400)


if __name__=='__main__':unittest.main()
