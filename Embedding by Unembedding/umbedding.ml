module type HLIST = sig 
  type 'a el
  type _ hlist = 
    | HNil : unit hlist 
    | HCons : 'a el * 'r hlist -> ('a * 'r) hlist

  val length : 'r hlist -> int 
end 

module HList (E: sig type 'a t end) : HLIST with type 'a el = 'a E.t = struct ()
  type 'a el = 'a E.t 
  type _ hlist = 
    | HNil : unit hlist 
    | HCons : 'a el * 'r hlist -> ('a * 'r) hlist 

  let rec length : type r. r hlist -> int = function
      HNil       -> 0 
    | HCons(_,r) -> 1 + length r 
end

module TEnv = HList(struct type 'a t = unit end)

module HL = HList( struct type 'a t = 'a end )

type (_,_,_) wapp =
  | AppNil  : (unit, 'b, 'b) wapp
  | AppStep : ('x , 'y , 'r) wapp -> ('a * 'x, 'y, 'a * 'r) wapp

module type TypeM = sig 
  type t 
end 

type (_,_) equal =
  | Refl : ('a,'a) equal 

let rec wapp_functional : 
  type xs ys zs1 zs2. 
  (xs,ys,zs1) wapp -> (xs,ys,zs2) wapp -> (zs1,zs2) equal = function
  | AppNil -> begin function AppNil -> Refl end  
  | AppStep w1 -> function AppStep w2 -> 
    match wapp_functional w1 w2 with  
    | Refl -> Refl 

module HListP(H: HLIST) = struct
  include H

  type (_,_) app_hlist =
  | AppHList : ('x,'y,'r) wapp * 'r hlist -> ('x ,'y) app_hlist  
       
  let rec append_hlist : type x y. x hlist -> y hlist -> (x,y) app_hlist = fun xs ys -> 
    match xs with
    | HNil -> AppHList (AppNil, ys)
    | HCons(a,r) -> 
      match append_hlist r ys with 
      | AppHList (w,res) -> AppHList(AppStep w, HCons(a,res))
end


module type VARIABLES = sig
  type (_, _) sem

  val var   : ('a, 'a * _) sem  
  val weaken : ('a , 'r) sem -> ('a , _ * 'r) sem
end

module Variables(H: VARIABLES) = struct
  include H

  let rec go : type a a' b. int -> a TEnv.hlist -> a' TEnv.hlist -> (b, a) sem -> (b, a') sem = fun n a b sem ->
    match n with
    | 0 -> Obj.magic sem
    | _ -> match b with 
           | TEnv.HNil -> failwith "Cannot happen"
           | TEnv.HCons (_, b') -> weaken (go (n-1) a b' sem)

  let weakenMany : type a a' b. a TEnv.hlist -> a' TEnv.hlist -> (b, a) sem -> (b, a') sem = fun a b sem ->
    let l1 = TEnv.length a in 
    let l2 = TEnv.length b in 
    let lenDiff = l2 - l1 in 
    go lenDiff a b sem

  type 'a envi = { runEnvI : 'e. 'e TEnv.hlist -> ('a, 'e) sem}

  let runOpen : type a b.(a envi -> (b envi)) -> (b, a * unit) sem = fun f ->
    let gA = TEnv.HCons ((), TEnv.HNil) in 
    let x  = { runEnvI = fun g' -> weakenMany gA g' var } in 
    (f x).runEnvI gA

  module HListEnvI = HList(struct type 'a t = 'a envi end)

  type ('xs, 'x) sig2 = |

  type (_,_) termRep = 
  | TR : ('xs,'env,'r) wapp * ('x, 'r) sem -> ('env, ('xs, 'x) sig2) termRep


  type (_) uRep =
  | UR : 'xs TEnv.hlist * ('xs HListEnvI.hlist -> 'x envi) -> ('xs,'x) sig2 uRep 

  module URepEnv = HList(struct type 'a t = 'a uRep end)

  module TermRepEnv(E: TypeM) = HList(struct type 'a t = (E.t,'a) termRep end)

  type ('env,_) termRepEnv_hlist =
  | THNil  : ('env,unit) termRepEnv_hlist
  | THCons : ('env,'s) termRep * ('env, 'ss) termRepEnv_hlist -> ('env,'s * 'ss) termRepEnv_hlist
 
  module MapToTermRepHList(H : HLIST) = struct 
    type 'env map_func = { f : 'a. 'a H.el -> ('env,'a) termRep  }
    let rec map : type r. 'env map_func -> r H.hlist -> ('env,r) termRepEnv_hlist = fun func l ->
      match l with 
      | H.HNil -> THNil
      | H.HCons(x,r) -> THCons(func.f x, map func r )   
  end

  type ('ss,'rr) semTerm = {f: 'env. ('env,'ss) termRepEnv_hlist -> ('rr, 'env) sem }

  let liftSO' : 
    type ss rr. (ss,rr) semTerm -> ss URepEnv.hlist -> rr envi = fun ff ks -> 
      { runEnvI = fun (type env) (e : env TEnv.hlist) -> 
        let module App = HListP(TEnv) in
        let module M_map = MapToTermRepHList(URepEnv) in  
        let rec mkXs : type env ys ys_env. env TEnv.hlist -> ys TEnv.hlist
            -> (ys, env, ys_env) wapp 
            -> ys_env TEnv.hlist -> ys HListEnvI.hlist = fun p ys wit te -> 
             match ys, wit, te with 
             | HNil, _, _ -> HNil 
             | (HCons(_,ys')),AppStep(wit'),HCons(_,te') ->
               let x = { runEnvI = fun e' -> weakenMany te e' H.var} in 
               HCons(x,mkXs p ys' wit' te') 
            in                       
        let cnv : type env xs x. env TEnv.hlist -> xs TEnv.hlist 
                   -> (xs HListEnvI.hlist -> x envi)
                   -> (env, (xs,x) sig2) termRep = fun e e1 k -> 
           match App.append_hlist e1 e with
           | AppHList(wit, ex_e) -> 
             let xs = mkXs e e1 wit ex_e in 
             TR(wit, (k xs).runEnvI ex_e) in 
        let conv : type env s. env TEnv.hlist -> s uRep -> (env,s) termRep =
         fun e ur -> match ur with 
         | UR(e1, k) -> cnv e e1 k in      
        ff.f (M_map.map {f = fun xs -> conv e xs} ks)
      }

  module CurryUncurry(H : HLIST) = struct
  
    type (_,_,_) cspec = 
    | Z : ('r, 'r, unit) cspec 
    | S : ('r, 'f, 'xs) cspec -> ('r, 'a H.el -> 'f, ('a * 'xs)) cspec 
  
    let rec uncurry : type r f xs. (r,f,xs) cspec -> f -> xs H.hlist -> r = 
      fun s h args -> match s, args with  
      | Z ,   _           -> h 
      | S ss, HCons(x,xs) -> uncurry ss (h x) xs   

    let rec cspec2TEnv : type r f xs. (r, f, xs) cspec -> xs TEnv.hlist = fun n -> match n with   
      | Z -> TEnv.HNil 
      | S s' -> TEnv.HCons ((), cspec2TEnv s')
  end
  
  module CU_EnvI = CurryUncurry(HListEnvI)

  let toURep : 
    type f xs x. 
    (x envi, f, xs) CU_EnvI.cspec
    -> f   
    -> (xs,x) sig2 uRep
    = fun cs f -> UR (CU_EnvI.cspec2TEnv cs, CU_EnvI.uncurry cs f)

  type (_,_,_) tf_spec =
    | SNil  : ('rr, 'rr, unit) tf_spec 
    | SCons : ('x envi, 'f, 'xs) CU_EnvI.cspec * ('rr, 'ff, 'ss) tf_spec -> ('rr, 'f -> 'ff, (('xs, 'x) sig2 * 'ss)) tf_spec 

  let rec toFuncU : 
    type rr ff ss. 
    (rr envi,ff,ss) tf_spec -> (ss URepEnv.hlist -> rr envi) -> ff
    = fun tfs h -> match tfs with
        | SNil -> h HNil 
        | SCons(cs,tfs) -> fun k -> toFuncU tfs (fun r -> h (HCons(toURep cs k,r)))

  type (_, _,_,_,_) ff_spec = 
    | FNil  : ('dummy, 'env,'rr, ('rr,'env) sem,unit) ff_spec 
    | FCons : ('xs,'env,'app) wapp  * ('dummy,'env,'rr,'ff,'ss) ff_spec 
              -> ('app * 'dummy, 'env, 'rr, ('x,'app) sem -> 'ff , (('xs,'x) sig2 * 'ss)) ff_spec
          
  type (_,_,_,_,_) wapp_cspec = 
    | XZ : (unit, 'app, 'app, 'r, 'r) wapp_cspec 
    | XS : ('xs,  'ys,  'app, 'r, 'f) wapp_cspec 
            -> (('x * 'xs), 'ys, ('x * 'app), 'r, 'x envi -> 'f) wapp_cspec 
          
  let rec to_wapp : type xs ys app r f. (xs, ys, app, r, f) wapp_cspec -> (xs,ys,app) wapp = 
    function
      | XZ   -> AppNil 
      | XS n -> AppStep (to_wapp n) 
          
  let rec to_cspec : type xs ys app r f. (xs, ys, app, r, f) wapp_cspec -> (r, f, xs) CU_EnvI.cspec = 
    function 
      | XZ  -> Z
      | XS n -> S (to_cspec n) 
          
  type (_,_,_,_,_,_) tf_ff_spec =
    | XNil  : ('dummy, 'env, 'rr, 'rr envi, ('rr,'env) sem, unit) tf_ff_spec 
    | XCons : ('xs, 'env, 'app, 'x envi, 'f) wapp_cspec * 
              ('dummy, 'env, 'rr, 'toFunc, 'fromFunc, 'ss) tf_ff_spec -> 
              ('app * 'dummy, 'env, 'rr, 'f -> 'toFunc, ('x,'app) sem -> 'fromFunc, (('xs,'x) sig2 * 'ss)) tf_ff_spec 
          
  let rec to_tf_spec : type dummy env rr toFunc fromFunc ss. (dummy, env, rr, toFunc, fromFunc, ss) tf_ff_spec -> (rr envi, toFunc, ss) tf_spec =
    function 
      | XNil -> SNil 
      | XCons(x, xs) -> SCons(to_cspec x, to_tf_spec xs) 
              
  let rec to_ff_spec : type dummy env rr toFunc fromFunc ss. (dummy, env, rr, toFunc, fromFunc, ss) tf_ff_spec -> (dummy, env, rr, fromFunc, ss) ff_spec =
    function
      | XNil -> FNil 
      | XCons(x,xs) -> FCons(to_wapp x, to_ff_spec xs) 
          
  let toF : type dummy env rr toFunc fromFunc ss. (dummy, env, rr, toFunc, fromFunc, ss) tf_ff_spec -> (ss URepEnv.hlist -> rr envi) -> toFunc = fun s -> toFuncU (to_tf_spec s)       
          
  let rec fromFunc : 
    type dummy env res tt ss. (dummy, env, res, tt, ss) ff_spec -> tt -> ((env, ss) termRepEnv_hlist -> (res,env) sem) = 
      function 
        | FNil -> fun x _ -> x 
        | FCons (w1, fs) -> fun f -> function 
            | THCons(TR(w2,t),ts) -> match wapp_functional w1 w2 with 
            | Refl -> fromFunc fs (f t) ts 
                      
  let fromF : type dummy env rr toFunc fromFunc ss. (dummy, env, rr, toFunc, fromFunc, ss) tf_ff_spec -> fromFunc -> ((env, ss) termRepEnv_hlist -> (rr,env) sem) = fun s -> fromFunc (to_ff_spec s) 

end
