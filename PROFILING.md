# Profiling

リポジトリのルートから通常の環境で実行します。

```bash
uv run --extra gpu --extra vision reactive-franka --headless --fast --steps 630 \
  --profile sync --profile-warmup-steps 30 --output outputs/profile-sync
```

profiling の出力は **`profile.json` 1 ファイルだけ**です。
各 stage の呼び出し数、平均・中央値・p95・p99、個々の時間 `samples_ms` を含みます。
既存のシミュレーション画像・summary.json・telemetry.jsonl は従来通り保存します。

- `--profile sync`: `gpu=True` の span の前後で CUDA の完了を待ちます。
  CPU 処理・転送・GPU 完了待ちを含む経過時間です。同期は実行タイミングに影響します。
- `--profile wall`: 追加の GPU 同期をせず、Python 呼び出しが戻るまでを測ります。
  非同期 GPU 処理の待ちが後段に移る場合があるため、GPU 単体の時間ではありません。
- `--profile` の省略: 計測・追加同期・profile.json の出力を無効にします。
- `--profile-warmup-steps N`: 最初の N 制御ステップを計測対象から外します。
  初期化と終了時の撮影も対象外です。warmup の選択は main 側で行います。
- `--fast`: 30 Hz に合わせる待機を除去します。外すと `pacing.sleep` も測ります。

## コードの使い方

```python
profiler = Profiler("sync")
with profiler.span("mapping.tsdf", gpu=True):
    mapper.integrate(observation)
profiler.save(output / "profile.json")
```

Profiler が持つ状態は mode と samples だけです。
span は前後の時計から時間を保存し、save は集計して JSON に保存します。
ファイルを保持しないため close はありません。
stage 名は解釈せず、文字列ごとに集計します。業務固有の分岐や属性辞書はありません。
認識・MPC の内部計測では `estimate(..., profiler=profiler)`、
`update_map(..., profiler=profiler)`、`command(..., profiler=profiler)` と明示的に渡します。

## 主な stage

| stage | 内容 |
|---|---|
| `camera.total` | シミュレーションの RGB-D 生成・変換 |
| `perception.total` | YOLO と点群処理・ICP・姿勢判定全体 |
| `yolo.preprocess` / `yolo.inference` / `yolo.postprocess` | 入力変換／Ultralytics predict 全体／候補選択・CPU マスク取得 |
| `pose.preprocess` / `icp.fit` / `pose.postprocess` | 点群・初期候補／全候補 ICP／品質判定・ハンドル方向 |
| `kalman.total@prediction` / `kalman.total@measurement` | 予測のみ／観測更新と予測 |
| `mapping.target_exclusion` | 推定位置周囲のカップ除外マスク生成 |
| `mapping.total` | depth フィルタ・転送・TSDF・ESDF・grid 反映 |
| `mapping.depth_filter` / `mapping.upload` | depth の除外処理／入力準備と GPU 転送 |
| `mapping.tsdf` / `mapping.esdf` / `mapping.publish` | TSDF 融合／ESDF 計算・取得／MPC の grid に反映 |
| `goal.total` | 目標生成とタスク遷移判定 |
| `mpc.total` | 関節状態取得から指令準備と有限性確認まで |
| `mpc.state_upload` / `mpc.goal_setup` | 関節状態取得・転送／FK と目標設定 |
| `mpc.optimize` / `mpc.validate` / `mpc.download` | 最適化／実行可能性確認・回復／CPU 指令取得 |
| `sim_step.total` | 物理更新と PyBullet 接触確認 |
| `step.total@perception` / `step.total@control` | 認識を行う／行わない制御ステップ全体 |

`--profile-warmup-steps 0` の場合、最初のステップは初期化時の観測を再利用するため
`step.total@initial` として区別します。
親の時間には子の時間が含まれます。両方を足さないでください。
全体の step 時間は、既存ログ保存・把持時の追加撮影・待機も含みます。
10 Hz / 30 Hz / 240 Hz はシミュレーション時間上の頻度です。
観測失敗時には ICP が実行されないため、各 stage の回数は異なります。
サンプルをメモリに保持するので、長時間計測は `--steps` や `--cycles` で区切ってください。
