use anyhow::Result;
use naga_oil::compose::{
    ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderDefValue,
};

pub struct ShaderCompiler {
    composer: Composer,
}

use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::{Path, PathBuf},
};

fn topological_depth_first(
    current: &str,
    graph: &HashMap<String, Vec<String>>,
    visited: &mut HashSet<String>,
    permanent: &mut HashSet<String>,
    sorted_nodes: &mut VecDeque<String>,
) -> Result<()> {
    if permanent.contains(current) {
        return Ok(());
    }

    if visited.contains(current) {
        anyhow::bail!("Cyclic dependency detected");
    }

    visited.insert(current.to_owned());

    for node in graph.get(current).unwrap_or(&vec![]) {
        topological_depth_first(node, graph, visited, permanent, sorted_nodes)?;
    }

    permanent.insert(current.to_owned());
    sorted_nodes.push_back(current.to_owned());
    Ok(())
}

fn construct_graphs(
    root: impl AsRef<Path>,
) -> (HashMap<String, PathBuf>, HashMap<String, Vec<String>>) {
    use std::fs;

    let mut traverse: Vec<PathBuf> = vec![PathBuf::from(root.as_ref())];
    let mut shader_files = vec![];

    while let Some(dir) = traverse.pop() {
        for entry in fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();

            let ftype = entry.file_type().unwrap();

            if ftype.is_file() {
                if let Some(extension) = entry.path().extension() {
                    if extension == "wgsl" {
                        shader_files.push(entry.path());
                    }
                }
            } else if ftype.is_dir() {
                traverse.push(entry.path());
            }
        }
    }

    let mut module_to_file = HashMap::new();
    let mut module_graph: HashMap<String, Vec<String>> = HashMap::new();

    for shader_file in shader_files {
        let contents = fs::read_to_string(&shader_file).unwrap();

        if let Some(module_name_pos) = contents.find("#define_import_path") {
            let module_name = contents[module_name_pos + "#define_import_path".len()..]
                .trim_start()
                .split_ascii_whitespace()
                .next()
                .unwrap();

            module_to_file.insert(module_name.to_owned(), shader_file);
            module_graph.entry(module_name.to_owned()).or_default();

            let mut pos = 0;
            while let Some(import_pos) = contents[pos..].find("#import ") {
                let import = contents[pos + import_pos + "#import ".len()..]
                    .split_terminator(';')
                    .next()
                    .unwrap();

                module_graph
                    .entry(module_name.to_owned())
                    .or_default()
                    .push(import.to_owned());
                pos += import_pos + "#import ".len();
            }
        }
    }

    for (module, imports) in module_graph.iter_mut() {
        for import in imports.iter_mut() {
            let proper_mod_name = module_to_file.keys().find(|x| import.starts_with(*x));

            if let Some(proper_mod_name) = proper_mod_name {
                *import = proper_mod_name.clone();
            } else {
                panic!("Module not found: import {} in {}", import, module)
            }
        }

        *imports = imports
            .iter()
            .cloned()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
    }

    (module_to_file, module_graph)
}

fn sorted_modules(graph: &HashMap<String, Vec<String>>) -> Result<Vec<String>> {
    let nodes = graph.keys().cloned().collect::<Vec<_>>();
    let mut sorted_nodes = VecDeque::new();
    let mut permanent_mark = HashSet::new();

    for node in nodes {
        topological_depth_first(
            &node,
            graph,
            &mut HashSet::new(),
            &mut permanent_mark,
            &mut sorted_nodes,
        )?;
    }

    Ok(sorted_nodes.into_iter().collect())
}

impl ShaderCompiler {
    pub fn new(module_repository: impl AsRef<Path>) -> Result<Self> {
        let mut composer = Composer::default();

        let (module_to_file, module_graph) = construct_graphs(module_repository);

        let files = sorted_modules(&module_graph)?
            .into_iter()
            .map(|module| module_to_file[&module].clone());

        for file in files {
            let content = std::fs::read_to_string(&file).unwrap();
            composer.add_composable_module(ComposableModuleDescriptor {
                source: &content,
                file_path: file.to_str().ok_or(anyhow::anyhow!("Invalid path"))?,
                language: naga_oil::compose::ShaderLanguage::Wgsl,
                ..Default::default()
            })?;
        }

        Ok(Self { composer })
    }

    pub fn compile(
        &mut self,
        path: &str,
        shader_defs: Vec<(String, ShaderDefValue)>,
    ) -> Result<wgpu::naga::Module> {
        use std::fs;

        let module = self
            .composer
            .make_naga_module(NagaModuleDescriptor {
                source: &fs::read_to_string(path)?,
                file_path: path,
                shader_type: naga_oil::compose::ShaderType::Wgsl,
                shader_defs: HashMap::from_iter(shader_defs),
                additional_imports: &[],
            })
            .inspect_err(|e| eprintln!("{}", e.emit_to_string(&self.composer)))?;

        Ok(module)
    }
}
