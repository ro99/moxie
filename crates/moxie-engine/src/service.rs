//! Typed pull stream: one active generation, no queue, cleanup before terminal
//! delivery and on disconnect/drop. Presentation clients never step a model.
use crate::{Cancel, GenerationEvent, GenerationRequest, Program, Session};
use moxie_memory::Ledger;
use moxie_types::{Error, Scope};

#[derive(Debug, PartialEq)]
pub enum StartError {
    Busy,
    Rejected(Error),
}
impl std::fmt::Display for StartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Busy => f.write_str("busy: one generation is already active"),
            Self::Rejected(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for StartError {}

#[derive(Debug)]
pub struct GenerationService<'ledger, 'program> {
    ledger: &'ledger mut Ledger,
    program: Program<'program>,
    active: Option<Session<'program>>,
}
impl<'ledger, 'program> GenerationService<'ledger, 'program> {
    pub fn new(ledger: &'ledger mut Ledger, program: Program<'program>) -> Self {
        Self {
            ledger,
            program,
            active: None,
        }
    }
    pub fn start(&mut self, request: GenerationRequest<'_>) -> Result<(), StartError> {
        if self.active.is_some() {
            return Err(StartError::Busy);
        }
        self.active = Some(
            Session::create(self.program, request, self.ledger).map_err(StartError::Rejected)?,
        );
        Ok(())
    }
    pub fn is_idle(&self) -> bool {
        self.active.is_none()
    }
    pub fn charged_bytes(&self) -> u64 {
        self.ledger.scope_committed(Scope::Host)
    }
    pub fn next_event(&mut self, cancel: &Cancel) -> Option<GenerationEvent> {
        let session = self.active.as_mut()?;
        let event = match session.step(cancel) {
            Ok(event) => event,
            Err(Error::Cancelled { .. }) => GenerationEvent::Cancelled {
                usage: session.usage(),
            },
            Err(error) => GenerationEvent::Failed {
                error,
                usage: session.usage(),
            },
        };
        if matches!(
            event,
            GenerationEvent::Finished { .. }
                | GenerationEvent::Cancelled { .. }
                | GenerationEvent::Failed { .. }
        ) {
            self.active.take().unwrap().close(self.ledger);
        }
        Some(event)
    }
}
impl Drop for GenerationService<'_, '_> {
    fn drop(&mut self) {
        if let Some(session) = self.active.take() {
            session.close(self.ledger);
        }
    }
}
