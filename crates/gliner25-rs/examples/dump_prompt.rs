// COGNEE-EVAL: dumps the token ids of the prompt the engine would build.
use anyhow::Result;
use gliner25_rs::{SchemaTask, SchemaTransformer};
fn main() -> Result<()> {
    let tok = std::env::args().nth(1).expect("tokenizer.json");
    let tf = SchemaTransformer::from_tokenizer_file(std::path::Path::new(&tok))?;
    let spec: serde_json::Value =
        serde_json::from_slice(&std::fs::read(std::env::args().nth(2).expect("scen"))?)?;
    let ets = spec["entity_types"].as_object().unwrap();
    let rts = spec["relation_types"].as_object().unwrap();
    // the scenarios file order is what serde_json (BTreeMap) gives; re-read raw
    // order from the file text instead
    let raw = std::fs::read_to_string(std::env::args().nth(2).unwrap())?;
    let order = |section: &str, keys: Vec<String>| -> Vec<String> {
        let start = raw.find(&format!("\"{section}\"")).unwrap();
        let mut v: Vec<(usize, String)> = keys
            .into_iter()
            .map(|k| (raw[start..].find(&format!("\"{k}\"")).unwrap_or(usize::MAX), k))
            .collect();
        v.sort();
        v.into_iter().map(|(_, k)| k).collect()
    };
    let enames = order("entity_types", ets.keys().cloned().collect());
    let rnames = order("relation_types", rts.keys().cloned().collect());
    let mut tasks = vec![SchemaTask::Entities(enames.clone())];
    let mut descs: Vec<Vec<(String, String)>> = vec![enames
        .iter()
        .map(|k| (k.clone(), ets[k].as_str().unwrap().to_string()))
        .collect()];
    for r in &rnames {
        tasks.push(SchemaTask::Relations(
            format!("{r}: {}", rts[r].as_str().unwrap()),
            vec!["head".into(), "tail".into()],
        ));
        descs.push(Vec::new());
    }
    let text = spec["scenarios"][0]["text"].as_str().unwrap();
    let rec = tf.transform_with_descriptions(text, &tasks, &descs)?;
    println!("{}", serde_json::json!({
        "input_ids": rec.input_ids,
        "query_markers": rec.query_markers().0,
        "word_first": rec.word_first_positions(),
    }));
    Ok(())
}
